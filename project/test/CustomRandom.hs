{-# LANGUAGE FlexibleContexts #-}
module CustomRandom where

import qualified Data.Vector.Unboxed as V
import qualified Control.Monad as M

import Data.Vector.Unboxed ( Vector )
import Control.Monad.State ( State, MonadState(put, get) )
import Data.Bits ( Bits((.|.), xor) )
import NetworkBuilder
    ( NetworkConfig(..),
      TransformerWeighting(TransformerWeighting, freqCisImag,
                           tokenEmbeddingTable, rmsAttWeight, wq, wk, wv, wo, rmsFfnWeight,
                           w1, w2, w3, rmsFinalWeight, freqCisReal),
      Matrix )

-- Define the state type
type StateRNG = Int

-- Define the state monad with StateRNG
type CustomRNG a = State StateRNG a

-- Generating and returning the next value
nextRandomValue :: CustomRNG Float
nextRandomValue = do
  -- Get the current state (custom random value)
  currentValue <- get
  -- Update the state (generate next custom random value)
  let randomValue = customRandom currentValue
  put randomValue
  return $ fromIntegral (randomValue `mod` 1000) / 1000.0

-- Getting the current value
getRandomValue :: CustomRNG Float
getRandomValue = do
    currentValue <- get
    return $ fromIntegral (currentValue `mod` 1000) / 1000.0

seedRandomValue :: Int -> CustomRNG ()
seedRandomValue seed = do
    put seed

customRandom :: Int -> Int
customRandom currentValue =
  let valueStep1 = (currentValue * 63) `mod` 0xC4CB7296
      valueStep2 = valueStep1 `xor` 0x1754FBF
      valueStep3 = (valueStep2 * 0xFF) `mod` 4294967296
      valueStep4 = valueStep3 `xor` 0x222F42CB
      valueStep5 = valueStep4 .|. 0x1234567890
  in ((valueStep5 + 14351514) * 32) `mod` 7777333

generateRandomArray :: Int -> CustomRNG [Float]
generateRandomArray size = M.replicateM size nextRandomValue

generateRandomArrays :: Int -> Int -> CustomRNG [[Float]]
generateRandomArrays nrows ncols = M.replicateM nrows (generateRandomArray ncols)

generateRandomVector :: Int -> CustomRNG (Vector Float)
generateRandomVector size = V.fromList <$> M.replicateM size nextRandomValue

generateRandomVectors :: Int -> Int -> CustomRNG [Vector Float]
generateRandomVectors count size = M.replicateM count (generateRandomVector size)

generateRandomMatrices :: Int -> Int -> Int -> CustomRNG [Matrix Float]
generateRandomMatrices count nrows ncols = M.replicateM count (generateRandomVectors nrows ncols)

buildRandomNetworkConfig :: Int -> Int -> Int -> Int -> Int -> CustomRNG NetworkConfig
buildRandomNetworkConfig nSteps numLayers nVocab headDim hiddenDimension = do
    let dimension = headDim * numLayers
    tokenEmbeddingTable' <- generateRandomVectors nVocab dimension
    attentionWeights <- generateRandomVectors numLayers dimension
    queryWeights <- generateRandomMatrices numLayers dimension dimension
    keyWeights <- generateRandomMatrices numLayers dimension dimension
    valueWeights <- generateRandomMatrices numLayers dimension dimension
    outputWeights <- generateRandomMatrices numLayers dimension dimension
    ffnWeights <- generateRandomVectors numLayers dimension
    w1' <- generateRandomMatrices numLayers hiddenDimension dimension
    w2' <- generateRandomMatrices numLayers dimension hiddenDimension
    w3' <- generateRandomMatrices numLayers hiddenDimension dimension
    finalWeights <- generateRandomVector dimension
    freqCisReal' <- generateRandomVectors nSteps (headDim `div` 2)
    freqCisImag' <- generateRandomVectors nSteps (headDim `div` 2)
    return NetworkConfig { tokenDim = dimension,
        headDimension = headDim,
        hiddenDim = hiddenDimension,
        numLayers = numLayers,
        numAttentionHeads = numLayers,
        numKeyValueHeads = numLayers,
        vocabSize = nVocab,
        seqLen = nSteps,
        weighting = TransformerWeighting
            { tokenEmbeddingTable = tokenEmbeddingTable',
            rmsAttWeight = attentionWeights,
            wq = queryWeights,
            wk = keyWeights,
            wv = valueWeights,
            wo = outputWeights,
            rmsFfnWeight = ffnWeights,
            w1 = w1',
            w2 = w2',
            w3 = w3',
            rmsFinalWeight = finalWeights,
            freqCisReal = freqCisReal',
            freqCisImag = freqCisImag'
            }
        }
