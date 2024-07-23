{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module InferenceSpec (spec) where

import Test.Hspec
    ( describe,
      it,
      shouldBe,
      shouldMatchList,
      shouldSatisfy,
      Spec,
      Expectation )
import NetworkBuilder
    ( NetworkConfig(numAttentionHeads, weighting),
      TransformerWeighting(tokenEmbeddingTable, rmsAttWeight, wq,
                           freqCisReal, freqCisImag) )
import Inference
    ( computeQKV,
      applyRotations,
      matrixVectorMult,
      splitVector,
      rmsNorm,
      computeDeltaFFN )
import CustomRandom
    ( CustomRNG, generateRandomVector, buildRandomNetworkConfig )

import qualified Data.Vector.Unboxed as V
import Control.Monad.State ( evalState )

spec :: Spec
spec = do
  describe "Inference on a small network" $ do
    let
      nVocab = 320
      headDim = 8
      numLayers = 3
      indexLayer = 2
      nSteps = 5
      hiddenDimension = 2

    it "builds a small network correctly" $ do
      let
        smallNetworkConfig :: CustomRNG NetworkConfig
        smallNetworkConfig = do
          buildRandomNetworkConfig nSteps numLayers nVocab headDim hiddenDimension

        network = evalState smallNetworkConfig 2
        freqCisRealRow = (freqCisReal (weighting network)) !! 2
        freqCisImagRow = (freqCisImag (weighting network)) !! 2
        rmsAttention = rmsAttWeight (weighting network)
        tokenMatrix = tokenEmbeddingTable (weighting network)

      (head tokenMatrix V.! 0) `shouldBe` 0.047
      (length tokenMatrix) `shouldBe` 320
      (V.length (head tokenMatrix)) `shouldBe` 24
      (tokenMatrix !! 319 V.! 23) `shouldBe` 0.828
      (length rmsAttention) `shouldBe` 3
      V.toList (rmsAttention !! 2) `shouldMatchList` [0.448, 0.975, 0.957, 0.775, 0.288, 0.913, 0.529, 0.169, 0.7,
                  0.511, 0.013, 0.952, 0.401, 0.661, 0.845, 0.121, 0.272, 0.256,
                  0.376, 0.958, 0.046, 0.471, 0.226, 0.462]
      (V.toList freqCisRealRow) `shouldMatchList` [0.828, 0.145, 0.344, 0.043]
      (V.toList freqCisImagRow) `shouldMatchList` [0.981, 0.754, 0.745, 0.609]

    it "computes RMS norm correctly" $ do
      let
        smallNetworkConfig :: CustomRNG (NetworkConfig, V.Vector Float)
        smallNetworkConfig = do
          n <- buildRandomNetworkConfig nSteps numLayers nVocab headDim hiddenDimension
          t <- generateRandomVector (headDim * numLayers)
          return (n, t)

        (network, token) = evalState smallNetworkConfig 2
        rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)

      V.length rba `shouldBe` 24
      token V.! 0 `shouldBe` 0.445
      token V.! 23 `shouldBe` 0.529
      ((rmsAttWeight (weighting network)) !! indexLayer) V.! 0 `shouldBe` 0.448
      ((rmsAttWeight (weighting network)) !! indexLayer) V.! 7 `shouldBe` 0.169
      rba V.! 0 `shouldBe` 0.3445728
      rba V.! 23 `shouldBe` 0.42241627

    it "applies small rotations" $ do
      let
        headVector = V.fromList [1.0, 2.0, 3.0, 4.0]
        freqCisRealRow = V.fromList [0.5, 0.2]
        freqCisImagRow = V.fromList [0.8, 0.3]
        result = applyRotations headVector freqCisRealRow freqCisImagRow
        expected = [-1.1,  1.8, -0.6,  1.7]
      (V.toList result) `shouldMatchList` expected

    it "applies big rotations" $ do
      let
        headVector = V.fromList [1.0, 2.0, 3.0, 4.0]
        freqCisRealRow = V.fromList [0.5, 0.2]
        freqCisImagRow = V.fromList [0.8, 0.3]
        result = applyRotations headVector freqCisRealRow freqCisImagRow
        expected = [-1.1,  1.8, -0.6,  1.7]
      (V.toList result) `shouldMatchList` expected

    it "computes Q, K and V" $ do
      let
        smallQKV :: CustomRNG (NetworkConfig, V.Vector Float)
        smallQKV = do
          n <- buildRandomNetworkConfig nSteps numLayers nVocab headDim hiddenDimension
          t <- generateRandomVector (headDim * numLayers)
          return (n, t)

        (network, token) = evalState smallQKV 2
        freqCisRealRow = ((freqCisReal (weighting network)) !! 2)
        freqCisImagRow = ((freqCisImag (weighting network)) !! 2)
        (qs, ks, vs) = computeQKV (weighting network) (numAttentionHeads network) indexLayer freqCisRealRow freqCisImagRow token
        rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)
        weightsQ = wq (weighting network)
        qVector = matrixVectorMult (weightsQ !! indexLayer) rba
        wQ = splitVector (numAttentionHeads network) (qVector)
        rotatedQ = applyRotations (wQ !! 2 ) freqCisRealRow freqCisImagRow

      (length (weightsQ !! indexLayer)) `shouldBe` 24
      (V.length (head (weightsQ !! indexLayer))) `shouldBe` 24
      (numAttentionHeads network) `shouldBe` 3
      (V.length qVector) `shouldBe` 24
      rba V.! 0 `shouldBe` 0.3445728
      rba V.! 23 `shouldBe` 0.42241627
      (length wQ) `shouldBe` 3
      (V.length (head wQ)) `shouldBe` 8
      (V.toList freqCisImagRow) `shouldMatchList` [0.981, 0.754, 0.745, 0.609]
      (V.toList freqCisRealRow) `shouldMatchList` [0.828, 0.145, 0.344, 0.043]

      (length qs) `shouldBe` 3
      (length ks) `shouldBe` 3
      (length vs) `shouldBe` 3
      (V.length rotatedQ) `shouldBe` 8

      shouldBeSmall 1e-3 $ vectorDistance (V.take 4 qVector) [4.652121 , 4.394577 , 5.8498507, 5.508383]
      shouldBeSmall 1e-3 $ vectorDistance (wQ !! 2) [5.5972257, 5.32329,   4.0131316, 5.037126,  4.0925946, 5.810919,  5.721209, 5.626199 ]
      shouldBeSmall 1e-3 $ vectorDistance rotatedQ [-0.58764449, 9.89856247, -3.21608903, 3.75628453, -2.92128194, 5.04793915, -3.18034321, 3.72614302]
      shouldBeSmall 1e-3 $ vectorDistance (qs !! 2) [-0.58764449, 9.89856247, -3.21608903, 3.75628453, -2.92128194, 5.04793915, -3.18034321, 3.72614302]
      shouldBeSmall 1e-3 $ vectorDistance (ks !! 1) [-1.262483, 9.873482, -1.809541, 4.85637, -1.716298, 4.831686, -2.449315, 3.406103]
      shouldBeSmall 1e-3 $ vectorDistance (vs !! 2) [4.61404 , 5.498788, 5.519291, 5.196641, 4.792354, 3.996622, 4.755136, 5.863463]

  describe "Delta FFN" $ do
    let
      nVocab = 32000
      headDim = 48
      nLayers = 6
      nSteps = 256
      hiddenDimension = 768

    it "computes delta FFN" $ do
      let
        networkForDeltaFFN :: CustomRNG (NetworkConfig, V.Vector Float)
        networkForDeltaFFN = do
          n <- buildRandomNetworkConfig nSteps nLayers nVocab headDim hiddenDimension
          t <- generateRandomVector 288
          return (n, t)
        (network, token) = evalState networkForDeltaFFN 2
        indexLayer = 4
        deltaFFN = computeDeltaFFN (weighting network) indexLayer token

      V.length deltaFFN `shouldBe` 288
      token V.! 0 `shouldBe` 0.616
      token V.! 287 `shouldBe` 0.176
      deltaFFN V.! 0 `shouldBe` 1749408.5
      deltaFFN V.! 287 `shouldBe` 1736454.9

      V.sum deltaFFN `shouldBe` 5.0058973e8
      V.minimum deltaFFN `shouldBe` 1723941.4
      V.maximum deltaFFN `shouldBe` 1753679.8

vectorDistance :: (V.Vector Float) -> [Float] -> Float
vectorDistance vector array = V.sum (V.zipWith (-) vector (V.fromList array))

shouldBeSmall :: Float -> Float -> Expectation
shouldBeSmall threshold a = shouldSatisfy a (\x -> abs (x) < threshold)
