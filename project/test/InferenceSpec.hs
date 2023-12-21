module InferenceSpec (spec) where

import Test.Hspec
import Inference
import CustomRandom

import qualified Data.Matrix as Mx
import qualified Data.Vector as V
import Control.Monad.State

spec :: Spec
spec = do
  describe "Helper functions" $ do
    it "replaces a value" $ do
      replaceAtIndex 1 3.0 [1.0, 2.0, 3.0] `shouldBe` [1.0, 3.0, 3.0]

    it "replacing works at the end of the list" $ do
      replaceAtIndex 2 0.5 [1.0, 2.0, 3.0] `shouldBe` [1.0, 2.0, 0.5]

  describe "Custom Random Values generator" $ do
    it "generates a custom random float" $ do
      let
        example :: CustomRNG Float
        example = do
          seedRandomValue 2
          -- run the random generator twice
          nextRandomValue
          nextRandomValue
          -- Get the current value
          randomValue <- getRandomValue
          -- Return a tuple of random value and counter value * 2
          return randomValue

        (result, finalState) = runState example 0

      result `shouldBe` 0.453
      finalState `shouldBe` 7460453

    it "generates custom random vector" $ do
      let result = evalState (generateRandomVector 3) 2
      result `shouldBe` (V.fromList [0.047, 0.453, 0.653])

    it "generates custom random matrix" $ do
      let result = evalState (generateRandomMatrix 3 4) 2
      (Mx.nrows result) `shouldBe` 3
      (Mx.ncols result) `shouldBe` 4

  describe "Inference on a small network" $ do
    let 
      nVocab = 320
      headDimension = 8
      nLayers = 3
      indexLayer = 2
      nSteps = 5
      hiddenDimension = 2

      smallQKV :: CustomRNG (Network, V.Vector Float)
      smallQKV = do
        n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
        t <- generateRandomVector (headDimension * nLayers)
        return (n, t)

      (network, token) = evalState smallQKV 2

    it "computes Q, K and V" $ do
      let
        freqCisRealRow = ((freqCisReal (weighting network)) !! 2)
        freqCisImagRow = ((freqCisImag (weighting network)) !! 2)
        (qs, ks, vs) = computeQKV network indexLayer freqCisRealRow freqCisImagRow token

      (length qs) `shouldBe` 3
      (V.toList freqCisImagRow) `shouldMatchList` [0.629, 0.403, 0.726, 0.048]
      (V.toList freqCisRealRow) `shouldMatchList` [0.171, 0.255, 0.385, 0.716]
      (V.toList (ks !! 1)) `shouldMatchList` ([-1.999097, 5.055076, -1.501158, 3.762684, -2.219901, 6.21974, 5.030888, 4.501329])
      (V.toList (vs !! 2)) `shouldMatchList` ([6.131495, 5.551599, 5.987549, 5.895988, 6.444849, 6.679024, 4.993975, 4.984156])
