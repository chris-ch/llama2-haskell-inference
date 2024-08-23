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

type LogitsVector s = AST.STArray s Int Float

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

buildActivation :: Int -> Vector Float -> M.Matrix Float -> Vector Float
buildActivation dimension headScores cacheLayerHead =
  DL.foldl' accumulate zeroVector zippedValues
  where
    accumulate :: Vector Float -> (Vector Float, Float) -> Vector Float
    accumulate acc (valueVector, attentionWeight) = V.zipWith (+) acc (scale attentionWeight valueVector)

    zeroVector = V.replicate dimension 0.0
    zippedValues = zip (M.getRowVectors cacheLayerHead) (V.toList headScores)
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

matrixVectorMult' :: LogitsVector s -> M.Matrix Float -> Vector Float -> TransformerStack s ()
matrixVectorMult' result mat vec = do
    lift $ forM_ (zip [0..] (M.getRowVectors mat)) $ \(i, row) -> do
        AST.writeArray result i (V.sum $ V.zipWith (*) row vec)

splitVector :: Int -> V.Vector Float -> [V.Vector Float]
splitVector m vec
  | m <= 0    = []
  | otherwise = go 0
  where
    len = V.length vec
    chunkSize = len `div` m
    go i
      | i >= len  = []
      | otherwise = V.slice i (min chunkSize (len - i)) vec : go (i + chunkSize)

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

splitEvery :: V.Unbox a => Int -> V.Vector a -> [V.Vector a]
splitEvery n vec
    | V.null vec = []
    | otherwise = chunk : splitEvery n rest
  where
    (chunk, rest) = V.splitAt n vec

computeQKV :: TransformerWeighting -> Int -> Int -> Int -> Vector Float -> Vector Float -> Vector Float -> (M.Matrix Float, M.Matrix Float, M.Matrix Float)
computeQKV weights numHeads dimHead indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token (rmsAttWeight weights !! indexLayer)
    wQ = M.multiplyVector (wq weights !! indexLayer) rba
    headsQ = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) (splitEvery dimHead wQ)
    wK = M.multiplyVector (wk weights !! indexLayer) rba
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) (splitEvery dimHead wK)
    headsV = M.Matrix numHeads dimHead $ M.multiplyVector (wv weights !! indexLayer) rba
  in
    (M.fromVectors numHeads dimHead headsQ, M.fromVectors numHeads dimHead headsK, headsV)

multiheadActivation :: forall s.  Int -> Int -> Int -> Int -> M.Matrix Float -> AttentionKV s -> TransformerStack s (M.Matrix Float)
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
    calculateScore indexHead keyVector = dotProduct (M.getRowVector headsQ indexHead) keyVector / sqrt (fromIntegral headDim)

  activations <- mapM (\indexHead -> do
      (k :: [Vector Float]) <- lift $ kVectors' indexHead
      (v :: [Vector Float]) <- lift $ vVectors' indexHead
      let
        scores :: Int -> Vector Float
        scores ixHead = softmax rawScores (indexToken + 1)
          where
            rawScores = V.generate (indexToken + 1) (\ixToken -> calculateScore ixHead (k !! ixToken))

        activation = buildActivation headDim (scores indexHead) (M.fromVectors numHeads numHeadComponents v)
      return activation
    ) [0 .. numHeads - 1]
  
  return $ M.fromVectors numHeads headDim activations

createTokenVectorForLayer :: forall s. Int -> Int -> Vector Float -> Vector Float -> TokenVector -> AttentionKV s -> TransformerStack s TokenVector
createTokenVectorForLayer indexToken indexLayer freqCisRealRow freqCisImagRow token sakv = do
    network <- ask

    let sKC = sKeyCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
    let sVC = sValueCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float

    let
      (headsQ, headsK, headsV) = computeQKV (weighting network) (numAttentionHeads network) (headDimension network) indexLayer freqCisRealRow freqCisImagRow token

      headsQ :: M.Matrix Float
      headsK :: M.Matrix Float
      headsV :: M.Matrix Float

    forM_ (zip [0..] (M.getRowVectors headsK)) $ \(i, headK) -> do
        forM_ [0..V.length headK - 1] $ \j -> do
            lift $ AST.writeArray sKC (indexToken, indexLayer, i, j) (headK V.! j)

    forM_ (zip [0..] (M.getRowVectors headsV)) $ \(i, headV) -> do
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

transformer :: forall s. LogitsVector s -> Int -> Token -> AttentionKV s -> TransformerStack s ()
transformer logits indexToken tokenCode sakv = do
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
    matrixVectorMult' logits (tokenEmbeddingTable (weighting network)) tokenWithRms

softmax' :: forall s. LogitsVector s -> Float -> Int -> TransformerStack s ()
softmax' values temperature size = do
    -- Find the maximum value in the relevant portion of the array
    maxVal <- lift $ do
        foldM (\acc i -> max acc <$> readArray values i) (negate (1/0)) [0..size - 1]

    -- Compute exponentials and store them in place
    sumExpValues <- lift $ do
        foldM (\acc i -> do
            x <- readArray values i
            let expVal = exp ((x - maxVal) / temperature)
            AST.writeArray values i expVal
            return (acc + expVal)
          ) 0 [0..size - 1]

    -- Normalize the values
    lift $ do
        forM_ [0..size - 1] $ \i -> do
            expVal <- readArray values i
            AST.writeArray values i (expVal / sumExpValues)

generateNextToken :: forall s. LogitsVector s -> Int -> Token -> Float -> Int -> AttentionKV s -> TransformerStack s Token
generateNextToken logits indexToken tokenCode temperature seedValue sakv = do
  network <- ask
  transformer logits indexToken tokenCode sakv
  nextToken <- case temperature of
    0.0 -> do
      logits' <- lift $ AST.getElems logits
      return $ fromIntegral (V.maxIndex (V.fromList logits'))
    _ -> do
        softmax' logits temperature (vocabSize network)
        logits' <- lift $ AST.getElems logits
        liftIO $ drawSample seedValue $ V.fromList logits'
  return nextToken

lookupNextToken :: Vocabulary -> Token -> Token -> BSC.ByteString
lookupNextToken vocab prevToken token = tokenStr
  where
    word = vocab !! fromIntegral token :: BS.ByteString
    firstChar = BSC.head word :: Char
    tokenStr = if prevToken == 1 && isSpace firstChar
          then BSC.tail (vocab !! fromIntegral token)
          else vocab !! fromIntegral token

generateTokens :: forall s. Int -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerStack s ([BSC.ByteString], Int)
generateTokens maxTokens promptTokens temperature vocab seedValue = do
  network <- ask
  let
    numHeads = numAttentionHeads network
    numHeadComponents = headDimension network

  sKC <- lift $ AST.newArray ((0, 0, 0, 0), (maxTokens - 1, numLayers network - 1, numHeads - 1, numHeadComponents - 1)) 0.0
  sVC <- lift $ AST.newArray ((0, 0, 0, 0), (maxTokens - 1, numLayers network - 1, numHeads - 1, numHeadComponents - 1)) 0.0

  let sakv = SAttentionKV { sKeyCache = sKC, sValueCache = sVC }

  mutableLogits <- lift $ AST.newArray (0 :: Int, vocabSize network - 1) (0.0 :: Float)
  go network mutableLogits 0 [] 1 sakv
    where
      go :: NetworkConfig -> LogitsVector s -> Int -> [BSC.ByteString] -> Token -> AttentionKV s -> TransformerStack s ([BSC.ByteString], Int)
      go network logits indexToken result token sakv
        | indexToken >= maxTokens || (indexToken /= 0 && token == 1) = return (result, indexToken)
        | indexToken < length promptTokens = do
          transformer logits indexToken token sakv
          let nextToken = promptTokens !! indexToken
          let nextTokenStr = lookupNextToken vocab token nextToken
          liftIO $ printf "%s" (BSC.unpack nextTokenStr)
          liftIO $ hFlush stdout
          go network logits (indexToken + 1) (result ++ [nextTokenStr]) nextToken sakv
        | otherwise = do
          nextToken <- generateNextToken logits indexToken token temperature seedValue sakv
          let nextTokenStr = lookupNextToken vocab token nextToken
          liftIO $ printf "%s" (BSC.unpack nextTokenStr)
          liftIO $ hFlush stdout
          go network logits (indexToken + 1) (result ++ [nextTokenStr]) nextToken sakv

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
