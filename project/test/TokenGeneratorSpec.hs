module TokenGeneratorSpec (spec) where

import Test.Hspec
import Builder
import Inference
import CustomRandom

import qualified Data.Vector.Unboxed as V
import Control.Monad.State
import Control.Monad (replicateM)

spec :: Spec
spec = do
  describe "Transformer" $ do
    let 
      nVocab = 32000
      headDim = 48
      nLayers = 6
      nSteps = 256
      hiddenDimension = 768
      seed = 2

    it "generates a token" $ do
      let

        networkForTokenGeneration :: CustomRNG (Network, V.Vector Float, [[[V.Vector Float]]], [[[V.Vector Float]]])
        networkForTokenGeneration = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDim hiddenDimension
          t <- generateRandomVector (headDim * nLayers)
          cK <- sequence [
            replicateM 6 (replicateM nLayers (generateRandomVector headDim)),
            replicateM 6 (replicateM nLayers (generateRandomVector headDim)),
            replicateM 2 (replicateM nLayers (generateRandomVector headDim))
            ]
          cV <- sequence [
            replicateM 6 (replicateM nLayers (generateRandomVector headDim)),
            replicateM 6 (replicateM nLayers (generateRandomVector headDim)),
            replicateM 2 (replicateM nLayers (generateRandomVector headDim))
            ]
          return (n, t, cK, cV)
        
        (network, token, cacheKey, cacheValue) = evalState networkForTokenGeneration seed
        tokenCode = 543
        stepCount = 2

      logits <- evalStateT (transformer tokenCode stepCount network) (RunCache {keyCache=cacheKey, valueCache=cacheValue})

      (V.take 5 logits) `shouldBe` V.fromList [76.487885,75.86574,69.82333,73.169655,72.23714]
      (V.take 5 (V.drop 31995 logits)) `shouldBe` V.fromList [81.34005,77.39803,78.24066,82.83305,75.38994]
