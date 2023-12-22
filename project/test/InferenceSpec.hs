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

    it "builds a small network correctly" $ do
      let
        smallNetwork :: CustomRNG Network
        smallNetwork = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
          return n
          
        network = evalState smallNetwork 2
        freqCisRealRow = (freqCisReal (weighting network)) !! 2
        freqCisImagRow = (freqCisImag (weighting network)) !! 2
        rmsAttention = rmsAttWeight (weighting network)
        tokenMatrix = tokenEmbeddingTable (weighting network)

      (Mx.getElem 1 1 tokenMatrix) `shouldBe` 0.047
      (Mx.nrows tokenMatrix) `shouldBe` 320
      (Mx.ncols tokenMatrix) `shouldBe` 24
      (Mx.getElem 320 24 tokenMatrix) `shouldBe` 0.828
      (length rmsAttention) `shouldBe` 3
      (V.toList (rmsAttention !! 2)) `shouldMatchList` [0.448, 0.975, 0.957, 0.775, 0.288, 0.913, 0.529, 0.169, 0.7,
                  0.511, 0.013, 0.952, 0.401, 0.661, 0.845, 0.121, 0.272, 0.256,
                  0.376, 0.958, 0.046, 0.471, 0.226, 0.462]
      (V.toList freqCisRealRow) `shouldMatchList` [0.828, 0.145, 0.344, 0.043]
      (V.toList freqCisImagRow) `shouldMatchList` [0.981, 0.754, 0.745, 0.609]

    it "computes RMS norm correctly" $ do
      let
        smallNetwork :: CustomRNG (Network, V.Vector Float)
        smallNetwork = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
          t <- generateRandomVector (headDimension * nLayers)
          return (n, t)
          
        (network, token) = evalState smallNetwork 2
        freqCisRealRow = ((freqCisReal (weighting network)) !! 2)
        freqCisImagRow = ((freqCisImag (weighting network)) !! 2)
        rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)

      V.length rba `shouldBe` 24
      token V.! 0 `shouldBe` 0.445
      token V.! 23 `shouldBe` 0.529
      ((rmsAttWeight (weighting network)) !! indexLayer) V.! 0 `shouldBe` 0.448
      ((rmsAttWeight (weighting network)) !! indexLayer) V.! 7 `shouldBe` 0.169
      rba V.! 0 `shouldBe` 0.3445728
      rba V.! 23 `shouldBe` 0.42241627
 
    it "computes Q, K and V" $ do
      let
        smallQKV :: CustomRNG (Network, V.Vector Float)
        smallQKV = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
          t <- generateRandomVector (headDimension * nLayers)
          return (n, t)

        (network, token) = evalState smallQKV 2
        freqCisRealRow = ((freqCisReal (weighting network)) !! 2)
        freqCisImagRow = ((freqCisImag (weighting network)) !! 2)
        (qs, ks, vs) = computeQKV network indexLayer freqCisRealRow freqCisImagRow token
        rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)
        weightsQ = wq (weighting network)
        qVector = matrixVectorMult (weightsQ !! indexLayer) rba
        wQ = splitVector (numAttentionHeads network) (qVector)
      
      (Mx.nrows (weightsQ !! indexLayer)) `shouldBe` 24
      (Mx.ncols (weightsQ !! indexLayer)) `shouldBe` 24
      (numAttentionHeads network) `shouldBe` 3
      (V.length qVector) `shouldBe` 24
      rba V.! 0 `shouldBe` 0.3445728
      rba V.! 23 `shouldBe` 0.42241627
      (length wQ) `shouldBe` 3
      (V.length (wQ !! 0)) `shouldBe` 8
      (V.toList freqCisImagRow) `shouldMatchList` [0.981, 0.754, 0.745, 0.609]
      (V.toList freqCisRealRow) `shouldMatchList` [0.828, 0.145, 0.344, 0.043]

      (length qs) `shouldBe` 3
      (length ks) `shouldBe` 3
      (length vs) `shouldBe` 3
      
      (take 4 (V.toList qVector)) `shouldMatchList` ([4.652121 , 4.394577 , 5.8498507, 5.508383])

      (V.toList (wQ !! 2)) `shouldMatchList` ([5.5972257, 5.32329,   4.0131316, 5.037126,  4.0925946, 5.810919,  5.721209, 5.626199 ])

      (V.toList (qs !! 2)) `shouldMatchList` ([-0.58764449,  9.89856247, -3.21608903,  3.75628453, -2.92128194, 5.04793915, -3.18034321,  3.72614302])
      (V.toList (ks !! 1)) `shouldMatchList` ([-1.262483, 9.873482, -1.809541, 4.85637, -1.716298, 4.831686,
            -2.449315, 3.406103])
      (V.toList (vs !! 2)) `shouldMatchList` ([4.61404 , 5.498788, 5.519291, 5.196641, 4.792354, 3.996622,
                  4.755136, 5.863463])
 