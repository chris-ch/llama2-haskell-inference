{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module HelperSpec (spec) where

import Test.Hspec ( describe, it, shouldBe, Spec )
import NetworkBuilder ( Matrix, readVectors )
import CustomRandom
    ( CustomRNG,
      nextRandomValue,
      getRandomValue,
      seedRandomValue,
      generateRandomVector,
      generateRandomVectors )

import qualified Data.Vector.Unboxed as V
import qualified Data.Binary.Get as BG
import qualified Data.Binary.Put as BP
import Control.Monad.State ( evalState, runState )

spec :: Spec
spec = do
  describe "Helper functions" $ do

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
          getRandomValue

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
      length result `shouldBe` 3
      V.length (head result) `shouldBe` 4
