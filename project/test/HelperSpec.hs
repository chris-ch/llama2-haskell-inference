module HelperSpec (spec) where

import Test.Hspec
import Builder
import Inference
import CustomRandom

import qualified Data.Vector.Unboxed as V
import qualified Data.Binary.Get as BG
import qualified Data.Binary.Put as BP
import Control.Monad.State

spec :: Spec
spec = do
  describe "Helper functions" $ do
    it "replaces a value" $ do
      replaceAtIndex 1 3.0 ([1.0, 2.0, 3.0]::[Float]) `shouldBe` ([1.0, 3.0, 3.0]::[Float])

    it "replaces at the end of the list" $ do
      replaceAtIndex 2 0.5 ([1.0, 2.0, 3.0]::[Float]) `shouldBe` ([1.0, 2.0, 0.5]::[Float])

    it "reshapes matrix as vector" $ do
      V.concat (fmap V.fromList [
            [0.047, 0.453, 0.653, 0.577],
             [0.022, 0.253, 0.432, 0.524],
             [0.114, 0.917, 0.747, 0.164]
        ]) `shouldBe` V.fromList ([0.047, 0.453, 0.653, 0.577, 0.022, 0.253, 0.432, 0.524, 0.114, 0.917, 0.747, 0.164]::[Float])

    it "reads a 2x3 matrix from a ByteString" $ do
        let inputBytes = BP.runPut $ mapM_ BP.putFloatle [1.0, 2.0, 3.0, 4.0, -1.0, -2.0]
            expectedMatrix :: Matrix Float
            expectedMatrix = fmap V.fromList [[1.0, 2.0, 3.0], [4.0, -1.0, -2.0]]
            actualMatrix = BG.runGet (readVectors 2 3) inputBytes
        actualMatrix `shouldBe` expectedMatrix

    it "reads a 3x2 matrix from a ByteString" $ do
        let inputBytes = BP.runPut $ mapM_ BP.putFloatle [1.0, 2.0, 3.0, 4.0, -1.0, -2.0]
            expectedMatrix :: Matrix Float
            expectedMatrix = fmap V.fromList [[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]]
            actualMatrix = BG.runGet (readVectors 3 2) inputBytes
        actualMatrix `shouldBe` expectedMatrix

  describe "Custom Random Values generator" $ do
    it "generates a custom random float" $ do
      let
        testRandom :: CustomRNG Float
        testRandom = do
          seedRandomValue 2
          -- run the random generator twice
          _ <- nextRandomValue
          _ <- nextRandomValue
          -- Get the current value
          randomValue <- getRandomValue
          -- Return a tuple of random value and counter value * 2
          return randomValue

        (result, finalState) = runState testRandom 0

      result `shouldBe` 0.453
      finalState `shouldBe` 7460453

    it "generates custom random vector(3)" $ do
      let result = evalState (generateRandomVector 3) 2
      V.length result `shouldBe` 3
      result `shouldBe` (V.fromList [0.047, 0.453, 0.653])
      
    it "generates custom random vector(4)" $ do
      let result = evalState (generateRandomVector 4) 2
      V.length result `shouldBe` 4
      result `shouldBe` (V.fromList [0.047, 0.453, 0.653, 0.577])
    
    it "generates custom random vectors consecutively" $ do
      let 
        vectorsTwice :: CustomRNG (V.Vector Float, V.Vector Float)
        vectorsTwice = do
          v1 <- generateRandomVector 4
          v2 <- generateRandomVector 4
          return (v1, v2)

        (vector1, vector2) = evalState (vectorsTwice) 2
      V.length vector1 `shouldBe` 4
      V.length vector2 `shouldBe` 4
      vector1 `shouldBe` (V.fromList [0.047, 0.453, 0.653, 0.577])
      vector2 `shouldBe` (V.fromList [0.022, 0.253, 0.432, 0.524])
    
    it "generates custom random matrix" $ do
      let result = evalState (generateRandomVectors 3 4) 2
      (length result) `shouldBe` 3
      (V.length (result !! 0)) `shouldBe` 4

    it "draws sample from CDF" $ do
        indexHighestCDF 0.05 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 0
        indexHighestCDF 0.2 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 1
        indexHighestCDF 0.35 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 1
        indexHighestCDF 0.4 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 2
        indexHighestCDF 0.45 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 2
        indexHighestCDF 0.8 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 4
        indexHighestCDF 0.85 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 4
        indexHighestCDF 0.95 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 5
        indexHighestCDF 1.0 (V.fromList [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]) `shouldBe` 5
        indexHighestCDF 0.1 (V.fromList [1.0]) `shouldBe` 0
