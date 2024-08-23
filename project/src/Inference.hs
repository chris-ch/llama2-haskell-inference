{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant return" #-}

module Inference (run, computeQKV, rmsNorm, splitVector,
computeDeltaFFN, createTokenVectorForLayer, multiheadActivation,
buildActivation, applyRotations, transformer,
softmax, drawSample
 ) where

import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified System.Random as R
import qualified Data.Vector.Unboxed as V

import NetworkBuilder (NetworkConfig(..), AttentionKV(..),
  TransformerWeighting(..), Vocabulary, PromptTokens, Token, TokenVector,
  initModel, tokenizerInit)
import qualified Data.Array.ST as AST
import Control.Monad.Trans (lift)
import System.IO (hFlush, stdout)
import Control.Monad (foldM, forM_)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Vector.Unboxed (Vector)
import Data.Array.MArray (readArray)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Control.Monad.Reader
    ( MonadIO(liftIO),
      ReaderT(runReaderT),
      MonadReader(ask) )
import GHC.IO.Handle (Handle)
import GHC.Unicode (isSpace)
import Control.Monad.ST.Trans (STT, runSTT)

import qualified Matrix as M

type TransformerStack s a = ReaderT NetworkConfig (STT s IO) a

type LogitsVector = Vector Float

softmax :: Vector Float -> Int -> Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (/ sumExpValues) expValues

drawSample :: Int -> Vector Float -> IO Token
drawSample seedValue probabilities = do
  let
    gen = R.mkStdGen seedValue
    (r, _) = R.random gen :: (Float, R.StdGen)

    indexHighestCDF :: Float -> Vector Float -> Int
    indexHighestCDF rand vec = min (V.ifoldl' (indexHighest rand) 0 cdf) (V.length vec - 1)
        where
          cdf = V.scanl1 (+) vec
          indexHighest :: Float -> Int -> Int -> Float -> Int
          indexHighest rand' acc i v = if v <= rand' then i + 1 else acc

  return $ fromIntegral $ indexHighestCDF r probabilities

buildActivation :: Int -> [Float] -> [Vector Float] -> Vector Float
buildActivation dimension headScores cacheLayerHead =
  DL.foldl' accumulate zeroVector zippedValues
  where
    accumulate :: Vector Float -> (Vector Float, Float) -> Vector Float
    accumulate acc (valueVector, attentionWeight) = V.zipWith (+) acc (scale attentionWeight valueVector)

    zeroVector = V.replicate dimension 0.0
    zippedValues = zip cacheLayerHead headScores
    scale w = V.map (w *)

applyRotations :: Vector Float -> Vector Float -> Vector Float -> Vector Float
applyRotations headVector freqCisRealRow freqCisImagRow =
  V.fromList $ concatMap applyRotation [0,2..V.length headVector - 2]
  where
    applyRotation :: Int -> [Float]
    applyRotation headItemIndex = [value * real - valueNext * imag, value * imag + valueNext * real]
      where
        real = freqCisRealRow V.! (headItemIndex `div` 2)
        imag = freqCisImagRow V.! (headItemIndex `div` 2)
        value = headVector V.! headItemIndex
        valueNext = headVector V.! (headItemIndex + 1)

splitVector :: Int -> Vector Float -> [Vector Float]
splitVector m vec = V.fromList <$> DLS.chunksOf (V.length vec `div` m) (V.toList vec)

dotProduct :: Vector Float -> Vector Float -> Float
dotProduct vec1 vec2 = V.sum $ V.zipWith (*) vec1 vec2

rmsNorm :: Vector Float -> Vector Float -> Vector Float
rmsNorm vector weights =
  let
    squareNorm :: Vector Float -> Float
    squareNorm = V.foldl' cumSumSquare 0.0
      where
        cumSumSquare :: Float -> Float -> Float
        cumSumSquare acc v = acc + v ^ (2::Int)

    ss = (squareNorm vector / fromIntegral (V.length vector)) + 1e-5
    normalized = V.map (* (1.0 / sqrt ss)) vector
  in V.zipWith (*) weights normalized

computeDeltaFFN :: TransformerWeighting -> Int -> TokenVector -> Vector Float
computeDeltaFFN weights indexLayer token =
    let
      sigmoidLinearUnit :: Float -> Float
      sigmoidLinearUnit value = value / (1.0 + exp (-value))

      rmsFFNWeight = rmsFfnWeight weights !! indexLayer :: Vector Float
      weight1 = w1 weights !! indexLayer :: M.Matrix Float
      weight2 = w2 weights !! indexLayer :: M.Matrix Float
      weight3 = w3 weights !! indexLayer :: M.Matrix Float
      rba = rmsNorm token rmsFFNWeight :: Vector Float
      hiddenDimensionBuffer1 = M.multiplyVector weight1 rba :: Vector Float
      hiddenDimensionBuffer2 = M.multiplyVector weight3 rba :: Vector Float
      sigmoided = V.map sigmoidLinearUnit hiddenDimensionBuffer1 :: Vector Float
    in
      M.multiplyVector weight2 (V.zipWith (*) sigmoided hiddenDimensionBuffer2)

computeQKV :: TransformerWeighting -> Int -> Int -> Vector Float -> Vector Float -> Vector Float -> ([Vector Float], [Vector Float], [Vector Float])
computeQKV weights numHeads indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token (rmsAttWeight weights !! indexLayer)
    wQ = splitVector numHeads (M.multiplyVector (wq weights !! indexLayer) rba)
    headsQ = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wQ
    wK = splitVector numHeads (M.multiplyVector (wk weights !! indexLayer) rba)
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wK
    headsV = splitVector numHeads (M.multiplyVector (wv weights !! indexLayer) rba)
  in
    (headsQ, headsK, headsV)

multiheadActivation :: forall s.  Int -> Int -> Int -> Int -> [Vector Float] -> AttentionKV s -> TransformerStack s (M.Matrix Float)
multiheadActivation indexToken numHeads headDim indexLayer headsQ sakv = do

  network <- ask

  let
    numHeadComponents = headDimension network
    sKC = sKeyCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
    sVC = sValueCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
  
  let
    kVectors' :: Int -> STT s IO [Vector Float]
    kVectors' ixHead = mapM (\ixToken -> do
        values <- V.generateM numHeadComponents (\ixComponent -> readArray sKC (ixToken, indexLayer, ixHead, ixComponent))
        return values
      ) [0..indexToken]

    vVectors' :: Int -> STT s IO [Vector Float]
    vVectors' ixHead = mapM (\ixToken -> do
        values <- V.generateM numHeadComponents (\ixComponent -> readArray sVC (ixToken, indexLayer, ixHead, ixComponent))
        return values
      ) [0..indexToken]

    calculateScore :: Int -> Vector Float -> Float
    calculateScore indexHead keyVector = dotProduct (headsQ !! indexHead) keyVector / sqrt (fromIntegral headDim)

  activations <- mapM (\indexHead -> do
      k <- lift $ kVectors' indexHead
      v <- lift $ vVectors' indexHead
      let
        scores :: Int -> [Float]
        scores ixHead = V.toList $ softmax rawScores (indexToken + 1)
          where
            rawScores = V.generate (indexToken + 1) (\ixToken -> calculateScore ixHead (k !! ixToken))

        activation = buildActivation headDim (scores indexHead) v
      return activation
    ) [0 .. numHeads - 1]
  
  return $ M.fromVectors numHeads headDim activations

createTokenVectorForLayer :: forall s. Int -> Int -> Vector Float -> Vector Float -> TokenVector -> AttentionKV s -> TransformerStack s TokenVector
createTokenVectorForLayer indexToken indexLayer freqCisRealRow freqCisImagRow token sakv = do
    network <- ask

    let sKC = sKeyCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
    let sVC = sValueCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float

    let
      (headsQ, headsK, headsV) = computeQKV (weighting network) (numAttentionHeads network) indexLayer freqCisRealRow freqCisImagRow token

      headsQ :: [Vector Float]
      headsK :: [Vector Float]
      headsV :: [Vector Float]

    forM_ (zip [0..] headsK) $ \(i, headK) -> do
        forM_ [0..V.length headK - 1] $ \j -> do
            lift $ AST.writeArray sKC (indexToken, indexLayer, i, j) (headK V.! j)

    forM_ (zip [0..] headsV) $ \(i, headV) -> do
        forM_ [0..V.length headV - 1] $ \j -> do
            lift $ AST.writeArray sVC (indexToken, indexLayer, i, j) (headV V.! j)

    activations' <- multiheadActivation indexToken (numAttentionHeads network) (headDimension network) indexLayer headsQ sakv

    let
      wO = wo (weighting network)
      deltaTokenQKV = M.multiplyVector (wO !! indexLayer) (M.values activations')
      token' = V.zipWith (+) token deltaTokenQKV :: TokenVector
      deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token' :: Vector Float
      result = V.zipWith (+) token' deltaTokenFFN :: TokenVector

    return result

transformer :: Int -> Token -> AttentionKV s -> TransformerStack s LogitsVector
transformer indexToken tokenCode sakv = do
    network <- ask

    -- Getting the token embedding
    let token = M.getRowVector (tokenEmbeddingTable (weighting network)) (fromIntegral tokenCode) :: TokenVector

    -- Plucking out the current row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = freqCisReal (weighting network) !! indexToken :: Vector Float
    let freqCisImagRow = freqCisImag (weighting network) !! indexToken :: Vector Float

    -- Forwarding all layers
    finalToken <- foldM (\accToken indexLayer -> createTokenVectorForLayer indexToken indexLayer freqCisRealRow freqCisImagRow accToken sakv)
                  token
                  [0..numLayers network - 1]

    -- Final rmsnorm
    let tokenWithRms = rmsNorm finalToken (rmsFinalWeight $ weighting network) :: TokenVector

    -- Classifier into logits
    let logits = M.multiplyVector (tokenEmbeddingTable (weighting network)) tokenWithRms :: LogitsVector

    return logits

generateNextToken :: Int -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> AttentionKV s -> TransformerStack s (BS.ByteString, Token)
generateNextToken indexToken promptTokens temperature vocab tokenCode seedValue sakv = do
  network <- ask
  logits <- transformer indexToken tokenCode sakv
  nextToken <- if indexToken < length promptTokens
    then return (promptTokens !! indexToken)
    else if temperature == 0.0
      then return $ fromIntegral (V.maxIndex logits)
    else do
      liftIO $ drawSample seedValue $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let
    word = vocab !! fromIntegral nextToken :: BS.ByteString
    firstChar = BSC.head word :: Char
    tokenStr = if tokenCode == 1 && isSpace firstChar
          then BSC.tail (vocab !! fromIntegral nextToken)
          else vocab !! fromIntegral nextToken
  return (tokenStr, nextToken)

generateTokens :: Int -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerStack s ([BSC.ByteString], Int)
generateTokens maxTokens promptTokens temperature vocab seedValue = do
  network <- ask

  let
    numHeads = numAttentionHeads network
    numHeadComponents = headDimension network

  sKC <- lift $ AST.newArray ((0, 0, 0, 0), (maxTokens - 1, numLayers network - 1, numHeads - 1, numHeadComponents - 1)) 0.0
  sVC <- lift $ AST.newArray ((0, 0, 0, 0), (maxTokens - 1, numLayers network - 1, numHeads - 1, numHeadComponents - 1)) 0.0

  let sakv = SAttentionKV { sKeyCache = sKC, sValueCache = sVC }

  go network 0 [] 1 sakv
    where
      go :: NetworkConfig -> Int -> [BSC.ByteString] -> Token -> AttentionKV s -> TransformerStack s ([BSC.ByteString], Int)
      go network indexToken result token sakv
        | indexToken >= maxTokens || (indexToken /= 0 && token == 1) = return (result, indexToken)
        | otherwise = do
          (tokenStr, nextToken) <- generateNextToken indexToken promptTokens temperature vocab token seedValue sakv
          liftIO $ printf "%s" (BSC.unpack tokenStr)
          liftIO $ hFlush stdout
          go network (indexToken + 1) (result ++ [tokenStr]) nextToken sakv

run :: Handle -> Handle -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileHandle tokenizerFileHandle temperature maxTokens prompt seed = do
  currentTime <- getPOSIXTime

  modelFileContent <- BS.hGetContents modelFileHandle
  tokenizerFileContent <- BS.hGetContents tokenizerFileHandle
  putStrLn "running inference..."

  let
    seedValue = fromMaybe (round currentTime) seed
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')

  printf "network: # layers %d\n" (numLayers config)
  printf "network: # attention heads %d / head dimension %d\n" (numAttentionHeads config) (headDimension config)
  printf "network: vocabulary size %d\n" $ vocabSize config
  printf "network: token dimensions %d\n" $ tokenDim config
  printf "prompt tokens: %s\n" $ show promptTokens
  printf "seed value %d, temperature %f\n" seedValue temperature
  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, countTokens) <- runSTT $ runReaderT (generateTokens maxTokens promptTokens temperature vocab seedValue) config

  endTime <- getPOSIXTime
  let
    duration :: Integer
    duration = round (endTime - startTime)
    tokensPerSec :: Float
    tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "duration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
