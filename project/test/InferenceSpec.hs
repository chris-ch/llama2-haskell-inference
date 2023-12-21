module InferenceSpec (spec) where

import Test.Hspec
import Inference (replaceAtIndex)
import CustomRandom (nextRandomValue, getRandomValue, seedRandomValue, generateRandomArray, CustomRNG)

import qualified Data.Matrix as M
import Control.Monad.State

spec :: Spec
spec = do
  describe "replaceAtIndex" $ do
    it "replaces a value" $ do
      replaceAtIndex 1 3.0 [1.0, 2.0, 3.0] `shouldBe` [1.0, 3.0, 3.0]

    it "works at the end of the list" $ do
      replaceAtIndex 2 0.5 [1.0, 2.0, 3.0] `shouldBe` [1.0, 2.0, 0.5]

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

    it "generates a custom random float" $ do
      let (result, finalState) = runState (generateRandomArray 3) 2
      result `shouldBe` [0.047, 0.453, 0.653]
