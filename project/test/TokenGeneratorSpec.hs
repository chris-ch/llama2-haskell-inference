{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module TokenGeneratorSpec (spec) where

import Test.Hspec
import NetworkBuilder
import Inference
import CustomRandom

import qualified Data.Vector.Unboxed as V
import Control.Monad.State ( evalState, evalStateT )
import Control.Monad (replicateM)
import Control.Monad.Reader ( ReaderT(runReaderT) )

spec :: Spec
spec = do
  describe "Transformer" $ do
    let 
      nVocab = 32000
      headDim = 48
      numLayers = 6
      nSteps = 256
      hiddenDimension = 768
      seed = 2

    it "generates a token" $ do
      let

        networkForTokenGeneration :: CustomRNG (NetworkConfig, V.Vector Float, [[[V.Vector Float]]], [[[V.Vector Float]]])
        networkForTokenGeneration = do
          n <- buildRandomNetworkConfig nSteps numLayers nVocab headDim hiddenDimension
          t <- generateRandomVector (headDim * numLayers)
          cK <- sequence [
            replicateM 6 (replicateM numLayers (generateRandomVector headDim)),
            replicateM 6 (replicateM numLayers (generateRandomVector headDim)),
            replicateM 2 (replicateM numLayers (generateRandomVector headDim))
            ]
          cV <- sequence [
            replicateM 6 (replicateM numLayers (generateRandomVector headDim)),
            replicateM 6 (replicateM numLayers (generateRandomVector headDim)),
            replicateM 2 (replicateM numLayers (generateRandomVector headDim))
            ]
          return (n, t, cK, cV)
        
        (network, _, cacheKey, cacheValue) = evalState networkForTokenGeneration seed
        tokenCode = 543
        stepCount = 2

      logits <- evalStateT (runReaderT (transformer tokenCode stepCount) network) (AttentionKV {keyCache=cacheKey, valueCache=cacheValue})

      (V.take 5 logits) `shouldBe` V.fromList [76.487885,75.86574,69.82333,73.169655,72.23714]
      (V.take 5 (V.drop 31995 logits)) `shouldBe` V.fromList [81.34005,77.39803,78.24066,82.83305,75.38994]

      next_token <- drawSample 11284652 $ softmax (V.map (/ 0.8) logits) (vocabSize network)
      next_token `shouldBe` 27569
