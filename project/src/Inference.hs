{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant return" #-}

module Inference (run, computeQKV, rmsNorm, splitVector,
computeDeltaFFN, createTokenVectorForLayer, multiheadActivation,
buildActivation, applyRotations, matrixVectorMult, transformer,
softmax, drawSample
 ) where

import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified System.Random as R
import qualified Data.Vector.Unboxed as V

import NetworkBuilder (NetworkConfig(..), AttentionKV(..), SAttentionKV(..),
  Matrix, TransformerWeighting(..), KeyCache, ValueCache, Vocabulary, PromptTokens, Token, TokenVector,
  initModel, tokenizerInit)
import Control.Monad.State ( StateT, evalStateT, MonadState(put), gets )
import qualified Data.Array.ST as AST
import Unsafe.Coerce
import Control.Monad.Trans (lift)
import System.IO (hFlush, stdout)
import Control.Monad (foldM, forM_)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Vector.Unboxed (Vector)
import Data.Array.MArray (readArray)  -- Add this import
import Data.Time.Clock.POSIX (getPOSIXTime)
import Control.Monad.Reader
    ( MonadIO(liftIO),
      ReaderT(runReaderT),
      MonadReader(ask) )
import GHC.IO.Handle (Handle)
import GHC.Unicode (isSpace)
import Control.Monad.ST.Unsafe (unsafeIOToST)
import Data.STRef (STRef, readSTRef, newSTRef, writeSTRef, modifySTRef')
import Control.Monad.ST.Trans (STT(..), runSTT, readSTArray)
import Control.Monad.ST (ST, runST, RealWorld, stToIO)
import qualified Data.Array.Unboxed as DAU
import GHC.Base (Any)

type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a
type TransformerResult' s a = ReaderT NetworkConfig (ST s) a

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

matrixVectorMult :: Matrix Float -> Vector Float -> Vector Float
matrixVectorMult mat vec = V.fromList $ map (`dotProduct` vec) mat

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
      weight1 = w1 weights !! indexLayer :: Matrix Float
      weight2 = w2 weights !! indexLayer :: Matrix Float
      weight3 = w3 weights !! indexLayer :: Matrix Float
      rba = rmsNorm token rmsFFNWeight :: Vector Float
      hiddenDimensionBuffer1 = matrixVectorMult weight1 rba :: Vector Float
      hiddenDimensionBuffer2 = matrixVectorMult weight3 rba :: Vector Float
      sigmoided = V.map sigmoidLinearUnit hiddenDimensionBuffer1 :: Vector Float
    in
      matrixVectorMult weight2 (V.zipWith (*) sigmoided hiddenDimensionBuffer2)

computeQKV :: TransformerWeighting -> Int -> Int -> Vector Float -> Vector Float -> Vector Float -> ([Vector Float], [Vector Float], [Vector Float])
computeQKV weights numHeads indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token (rmsAttWeight weights !! indexLayer)
    wQ = splitVector numHeads (matrixVectorMult (wq weights !! indexLayer) rba)
    headsQ = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wQ
    wK = splitVector numHeads (matrixVectorMult (wk weights !! indexLayer) rba)
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wK
    headsV = splitVector numHeads (matrixVectorMult (wv weights !! indexLayer) rba)
  in
    (headsQ, headsK, headsV)

computeScores :: Int -> Int -> Int -> [Vector Float] -> KeyCache -> [Float]
computeScores headDim indexLayer indexHead headsQ = map calculateScore
  where
    calculateScore :: [[Vector Float]] -> Float
    calculateScore keyVectors =
      let keyVector = ((keyVectors !! indexLayer) !! indexHead)
      in dotProduct (headsQ !! indexHead) keyVector / sqrt (fromIntegral headDim)

multiheadActivation'' :: Int -> Int -> Int -> KeyCache-> ValueCache -> [Vector Float] -> Matrix Float
multiheadActivation'' numHeads headDim indexLayer kC vC headsQ =
    [buildActivation headDim (scores indexHead) (cacheLayerHead indexHead) | indexHead <- [0 .. numHeads - 1]]
    where
      cacheLayerHead :: Int -> [Vector Float]
      cacheLayerHead index = map (\count -> vC !! count !! indexLayer !! index) [0..]

      scores :: Int -> [Float]
      scores index = V.toList $ softmax rawScores (V.length rawScores)
        where
          rawScores = V.fromList $ computeScores headDim indexLayer index headsQ kC

multiheadActivation :: forall s.  Int -> Int -> Int -> Int -> KeyCache -> ValueCache -> [Vector Float] -> SAttentionKV s -> ReaderT NetworkConfig (StateT AttentionKV (STT s IO)) (Matrix Float)
multiheadActivation indexToken numHeads headDim indexLayer kC' vC' headsQ sakv = do

  network <- ask

  let
    numHeadComponents = headDimension network
    sKC = sKeyCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
    sVC = sValueCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
  
  ((_, _, _, _), (maxTokenIndex, maxLayerIndex, maxHeadIndex, maxHeadComponentIndex)) <- lift . lift $ AST.getBounds sKC

{-   liftIO $ printf "max token index: %d\n" maxTokenIndex
  liftIO $ printf "max layer index: %d\n" maxLayerIndex
  liftIO $ printf "max head index: %d\n" maxHeadIndex
  liftIO $ printf "max head component index: %d\n" maxHeadComponentIndex
  liftIO $ printf "kC length: %d\n" $ length kC
  liftIO $ printf "index token: %d\n----\n" $ indexToken
 -}

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
      k <- lift . lift $ kVectors' indexHead
      v <- lift . lift $ vVectors' indexHead
      let
        scores :: Int -> [Float]
        scores ixHead = V.toList $ softmax rawScores (indexToken + 1)
          where
            rawScores = V.generate (indexToken + 1) (\ixToken -> calculateScore ixHead (k !! ixToken))

        activation = buildActivation headDim (scores indexHead) v
      return activation
    ) [0 .. numHeads - 1]
  
  return activations
    --where
      --vVectors :: Int -> [Vector Float]
      --vVectors indexHead = map (\ixToken -> vC !! ixToken !! indexLayer !! indexHead) [0..indexToken]
      --vVectors indexHead = [ vC !! ixToken !! indexLayer !! indexHead | ixToken <- [0..indexToken] ]

      --kVectors :: Int -> [Vector Float]
      --kVectors indexHead = map (\ixToken -> kC !! ixToken !! indexLayer !! indexHead) [0..indexToken]
      --kVectors indexHead = [ kC !! ixToken !! indexLayer !! indexHead | ixToken <- [0..indexToken] ]

createTokenVectorForLayer :: forall s. Int -> Int -> Vector Float -> Vector Float -> TokenVector -> SAttentionKV s -> ReaderT NetworkConfig (StateT AttentionKV (STT s IO)) TokenVector
createTokenVectorForLayer indexToken indexLayer freqCisRealRow freqCisImagRow token sakv = do
    network <- ask

    let sKC = sKeyCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float
    let sVC = sValueCache sakv :: AST.STUArray s (Int, Int, Int, Int) Float

    (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
    let
      (headsQ, headsK, headsV) = computeQKV (weighting network) (numAttentionHeads network) indexLayer freqCisRealRow freqCisImagRow token

      headsQ :: [Vector Float]
      headsK :: [Vector Float]
      headsV :: [Vector Float]
      keyCacheStep = (kC !! indexToken) ++ [headsK]
      valueCacheStep = (vC !! indexToken) ++ [headsV]
      keyCache' = take indexToken kC ++ [keyCacheStep]
      valueCache' = take indexToken vC ++ [valueCacheStep]

    -- Update key cache
    forM_ (zip [0..] headsK) $ \(i, headK) -> do
        forM_ [0..V.length headK - 1] $ \j -> do
            lift . lift $ AST.writeArray sKC (indexToken, indexLayer, i, j) (headK V.! j)

    -- Update value cache
    forM_ (zip [0..] headsV) $ \(i, headV) -> do
        forM_ [0..V.length headV - 1] $ \j -> do
            lift . lift $ AST.writeArray sVC (indexToken, indexLayer, i, j) (headV V.! j)

    -- Freeze the arrays to get immutable versions
    --frozenKC <- AST.freeze sKC
    --frozenVC <- AST.freeze sVC

    activations' <- multiheadActivation indexToken (numAttentionHeads network) (headDimension network) indexLayer keyCache' valueCache' headsQ sakv

    let
      --activations = multiheadActivation (numAttentionHeads network) (headDimension network) indexLayer keyCache' valueCache' headsQ
      wO = wo (weighting network)
      deltaTokenQKV = matrixVectorMult (wO !! indexLayer) (V.concat activations')
      token' = V.zipWith (+) token deltaTokenQKV :: TokenVector
      deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token' :: Vector Float
      result = V.zipWith (+) token' deltaTokenFFN :: TokenVector

    put (AttentionKV {keyCache = keyCache', valueCache = valueCache'})
    return result

transformer :: Int -> Token -> SAttentionKV s -> ReaderT NetworkConfig (StateT AttentionKV (STT s IO)) LogitsVector
transformer indexToken tokenCode sakv = do
    network <- ask

    -- Getting the token embedding
    let token = tokenEmbeddingTable (weighting network) !! fromIntegral tokenCode :: TokenVector

    -- Plucking out the current row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = freqCisReal (weighting network) !! indexToken :: Vector Float
    let freqCisImagRow = freqCisImag (weighting network) !! indexToken :: Vector Float

    -- Forwarding all the layers
    finalToken <- foldM (\accToken indexLayer -> createTokenVectorForLayer indexToken indexLayer freqCisRealRow freqCisImagRow accToken sakv)
                  token
                  [0..nLayers network - 1]

    -- Final rmsnorm
    let tokenWithRms = rmsNorm finalToken (rmsFinalWeight $ weighting network) :: TokenVector

    -- Classifier into logits
    let logits = matrixVectorMult (tokenEmbeddingTable (weighting network)) tokenWithRms :: LogitsVector

    return logits

generateNextToken :: Int -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> SAttentionKV s -> ReaderT NetworkConfig (StateT AttentionKV (STT s IO)) (BS.ByteString, Token)
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

generateTokens :: Int -> PromptTokens -> Float -> Vocabulary -> Int -> SAttentionKV s -> ReaderT NetworkConfig (StateT AttentionKV (STT s IO)) ([BS.ByteString], Int)
generateTokens maxTokens promptTokens temperature vocab seedValue sakv = do
  network <- ask
  go network 0 [] 1 where
    go network indexToken result token
      | indexToken >= maxTokens || (indexToken /= 0 && token == 1) = return (result, indexToken)
      | otherwise = do
        (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
        put (AttentionKV {keyCache = take indexToken kC ++ [[]], valueCache = take indexToken vC ++ [[]]})
        (tokenStr, nextToken) <- generateNextToken indexToken promptTokens temperature vocab token seedValue sakv
        liftIO $ printf "%s" (BSC.unpack tokenStr)
        liftIO $ hFlush stdout
        go network (indexToken + 1) (result ++ [tokenStr]) nextToken

generateTokens'' :: Int -> PromptTokens -> Float -> Vocabulary -> Int -> ReaderT NetworkConfig (StateT AttentionKV (STT s IO)) ([BSC.ByteString], Int)
generateTokens'' maxTokens promptTokens temperature vocab seedValue = do
  network <- ask

  let
    numLayers = nLayers network
    numHeads = numAttentionHeads network
    numHeadComponents = headDimension network
  sKC <- lift . lift $ AST.newArray ((0, 0, 0, 0), (maxTokens - 1, numLayers - 1, numHeads - 1, numHeadComponents - 1)) 0.0
  sVC <- lift . lift $ AST.newArray ((0, 0, 0, 0), (maxTokens - 1, numLayers - 1, numHeads - 1, numHeadComponents - 1)) 0.0

  let sakv = SAttentionKV { sKeyCache = sKC, sValueCache = sVC }

  -- Modify the arrays
  lift . lift $ forM_ [1..maxTokens] $ \i -> do
      AST.writeArray (sKeyCache sakv) (i - 1, 0, 0, 0) $ fromIntegral (i * i)

  lift . lift $ forM_ [1..numLayers] $ \i -> do
      AST.writeArray (sValueCache sakv) (0, i - 1, 0, 0) $ fromIntegral (i * i)

  -- Optional: read a value (for demonstration)
  value <- lift . lift $ readArray (sValueCache sakv) (3, 4, 0, 0)
  generateTokens maxTokens promptTokens temperature vocab seedValue sakv

    -- Freeze the mutable arrays to immutable arrays
    --frozen1 <- AST.freeze sKC
    --frozen2 <- AST.freeze sVC

    --return (frozen1, frozen2)

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
    initStateAttentionKV :: AttentionKV
    initStateAttentionKV = AttentionKV { keyCache = [], valueCache = [] }
  printf "network: # layers %d\n" (nLayers config)
  printf "network: # attention heads %d / head dimension %d\n" (numAttentionHeads config) (headDimension config)
  printf "network: vocabulary size %d\n" $ vocabSize config
  printf "network: token dimensions %d\n" $ dim config
  printf "prompt tokens: %s\n" $ show promptTokens
  printf "seed value %d, temperature %f\n" seedValue temperature
  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, countTokens) <- runSTT $ evalStateT (runReaderT (generateTokens'' maxTokens promptTokens temperature vocab seedValue) config) initStateAttentionKV

  endTime <- getPOSIXTime
  let
    duration :: Integer
    duration = round (endTime - startTime)
    tokensPerSec :: Float
    tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "duration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
