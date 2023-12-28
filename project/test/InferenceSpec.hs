module InferenceSpec (spec) where

import Test.Hspec
import Inference
import CustomRandom

import qualified Data.Matrix as Mx
import qualified Data.Vector as V
import Control.Monad.State
import Control.Monad (replicateM)

spec :: Spec
spec = do
  describe "Helper functions" $ do
    it "replaces a value" $ do
      replaceAtIndex 1 3.0 [1.0, 2.0, 3.0] `shouldBe` [1.0, 3.0, 3.0]

    it "replaces at the end of the list" $ do
      replaceAtIndex 2 0.5 [1.0, 2.0, 3.0] `shouldBe` [1.0, 2.0, 0.5]

    it "reshapes matrix as vector" $ do
      reshapeMatrixToVector (Mx.fromLists [
            [0.047, 0.453, 0.653, 0.577],
             [0.022, 0.253, 0.432, 0.524],
             [0.114, 0.917, 0.747, 0.164]
        ]) `shouldBe` V.fromList [0.047, 0.453, 0.653, 0.577, 0.022, 0.253, 0.432, 0.524, 0.114, 0.917, 0.747, 0.164]

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
        rotatedQ = applyRotations (wQ !! 2 ) freqCisRealRow freqCisImagRow
      
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
      (V.length rotatedQ) `shouldBe` 8
      
      shouldBeSmall 1e-3 $ vectorDistance (V.take 4 qVector) [4.652121 , 4.394577 , 5.8498507, 5.508383]
      shouldBeSmall 1e-3 $ vectorDistance (wQ !! 2) [5.5972257, 5.32329,   4.0131316, 5.037126,  4.0925946, 5.810919,  5.721209, 5.626199 ]
      shouldBeSmall 1e-3 $ vectorDistance rotatedQ [-0.58764449, 9.89856247, -3.21608903, 3.75628453, -2.92128194, 5.04793915, -3.18034321, 3.72614302]
      shouldBeSmall 1e-3 $ vectorDistance (qs !! 2) [-0.58764449, 9.89856247, -3.21608903, 3.75628453, -2.92128194, 5.04793915, -3.18034321, 3.72614302]
      shouldBeSmall 1e-3 $ vectorDistance (ks !! 1) [-1.262483, 9.873482, -1.809541, 4.85637, -1.716298, 4.831686, -2.449315, 3.406103]
      shouldBeSmall 1e-3 $ vectorDistance (vs !! 2) [4.61404 , 5.498788, 5.519291, 5.196641, 4.792354, 3.996622, 4.755136, 5.863463]
 
  describe "Multihead activation" $ do
    let 
      nVocab = 320
      headDimension = 48
      nLayers = 4
      indexLayer = 2
      nSteps = 5
      hiddenDimension = 2

    it "builds a network for activation" $ do
      let
        networkForActivation :: CustomRNG (Network, [V.Vector Float], [[[V.Vector Float]]], [[[V.Vector Float]]])
        networkForActivation = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
          hQ <- replicateM 6 (generateRandomVector 48)
          cK <- sequence [
            replicateM 6 (replicateM 6 (generateRandomVector 48)),
            replicateM 3 (replicateM 6 (generateRandomVector 48))
            ]
          cV <- sequence [
            replicateM 6 (replicateM 6 (generateRandomVector 48)),
            replicateM 3 (replicateM 6 (generateRandomVector 48))
            ]
          return (n, hQ, cK, cV)
        indexLayer = 2
        (network, headsQ, cacheKey, cacheValue) = evalState networkForActivation 2
        headScoresExample = [0.5194815185588364, 0.48051848144116366]
        activation = buildActivation headDimension indexLayer cacheValue 3 headScoresExample
        
        expectedActivation = [0.3045971,0.28449363,0.48838997,0.26805186,0.7258309,0.5840917,
          0.678182,0.6833122,0.7507793,0.48202664,0.26566213,0.4568178,0.32925987,0.72464937,
          0.7884679,0.5520643,0.5221176,0.27327257,0.3940515,0.1524674,0.38288274,0.90151936,
          0.44484353,0.617415,0.39233693,0.778013,0.5751566,0.5121434,0.5486367,0.83911717,
          0.722545,0.30416897,0.86215585,0.49119535,0.40411735,0.25259772,0.4708447,0.42280445,
          0.4961695,0.61828625,0.4113124,0.8776885,0.84770113,0.74740267,0.6527272,0.5420901,
          0.2864671,0.47077942]
       

        result = multiheadActivation network indexLayer cacheKey cacheValue headsQ

      activation `shouldBe` V.fromList expectedActivation

      (headsQ !! 0) `shouldBe` V.fromList [0.734,0.616,0.897,0.159,0.346,0.646,0.22,0.586,0.981,
          0.769,0.913,0.77,0.791,0.171,0.255,0.385,0.716,0.948,0.233,0.858,0.206,0.161,9.0e-2,
          0.195,0.828,0.145,0.344,4.3e-2,0.766,0.949,0.75,0.7,0.953,0.514,0.37,0.866,0.755,0.629,
          0.403,0.726,4.8e-2,0.821,0.872,0.752,0.981,0.754,0.745,0.609]
          
      (headsQ !! 5) `shouldBe` V.fromList [9.0e-2,0.195,0.828,0.145,0.344,4.3e-2,0.766,0.949,0.75,
        0.7,0.953,0.514,0.37,0.866,0.755,0.629,0.403,0.726,4.8e-2,0.821,0.872,0.752,0.981,0.754,
        0.745,0.609,0.162,7.6e-2,0.564,0.644,0.398,0.813,0.421,0.665,0.445,0.391,0.504,0.73,0.434,
        0.32,0.323,0.323,0.483,0.502,0.984,0.14,9.0e-2,0.232]

      (numAttentionHeads network) `shouldBe` 4
      (Mx.ncols result) `shouldBe` 48
      (Mx.nrows result) `shouldBe` 4
      (Mx.getRow 1 result) `shouldBe` V.fromList [0.30273914,0.6412468,0.4341167,0.313628,0.6088015,
        0.7288631,7.149603e-2,0.554964,0.32315883,0.43760967,0.8307215,0.3190574,0.35306537,0.5871704,
        0.64360785,0.87371045,0.15746486,0.6745846,0.36556137,0.3270446,0.44000852,0.40689552,0.17859621,
        0.9115449,0.26830727,0.6173085,0.62384546,0.44949543,0.20511425,0.31641296,0.53728104,0.58635247,
        0.41710815,0.5492132,0.5879383,0.2985614,0.28704336,0.49492365,0.26605985,0.72003424,0.6005455,
        0.6819469,0.69283384,0.75157607,0.49483508,0.26173794,0.44845143,0.33157054]

      shouldBeSmall 1e-3 $ vectorDistance (Mx.getRow 4 result) [0.3045971, 0.28449363, 0.48838997, 0.26805186, 0.72583091,
                                0.58409174, 0.67818201, 0.68331219, 0.7507793, 0.48202663,
                                0.26566214, 0.45681779, 0.32925986, 0.72464937, 0.78846788,
                                0.55206428, 0.5221176, 0.27327259, 0.3940515, 0.15246741,
                                0.38288274, 0.90151936, 0.44484355, 0.61741503, 0.39233694,
                                0.77801296, 0.57515665, 0.51214337, 0.54863667, 0.83911714,
                                0.72254506, 0.30416898, 0.86215585, 0.49119536, 0.40411736,
                                0.25259773, 0.47084469, 0.42280443, 0.49616951, 0.61828625,
                                0.41131239, 0.87768853, 0.84770113, 0.74740264, 0.65272719,
                                0.54209012, 0.28646711, 0.47077943]
          
  describe "Delta FFN" $ do
    let 
      nVocab = 32000
      headDimension = 48
      nLayers = 6
      nSteps = 256
      hiddenDimension = 768

    it "computes delta FFN" $ do
      let
        networkForDeltaFFN :: CustomRNG (Network, V.Vector Float)
        networkForDeltaFFN = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
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

    it "creates new token" $ do
      let
        networkForNewToken :: CustomRNG (Network, V.Vector Float, [[[V.Vector Float]]], [[[V.Vector Float]]])
        networkForNewToken = do
          n <- buildRandomNetwork nSteps nLayers nVocab headDimension hiddenDimension
          t <- generateRandomVector 288
          cK <- sequence [
            replicateM 6 (replicateM 6 (generateRandomVector 48)),
            replicateM 6 (replicateM 6 (generateRandomVector 48)),
            replicateM 2 (replicateM 6 (generateRandomVector 48))
            ]
          cV <- sequence [
            replicateM 6 (replicateM 6 (generateRandomVector 48)),
            replicateM 6 (replicateM 6 (generateRandomVector 48)),
            replicateM 2 (replicateM 6 (generateRandomVector 48))
            ]
          return (n, t, cK, cV)
        (network, token, cacheKey, cacheValue) = evalState networkForNewToken 2
        indexLayer = 2
        stepCount = 2
        freqCisRealRow = ((freqCisReal (weighting network)) !! 2)
        freqCisImagRow = ((freqCisImag (weighting network)) !! 2)

        (q, k, v) = computeQKV network indexLayer freqCisRealRow freqCisImagRow token
        (token', cacheKey', cacheValue') = createLayerToken network stepCount cacheKey cacheValue indexLayer freqCisRealRow freqCisImagRow token
        activations = multiheadActivation network indexLayer cacheKey' cacheValue' q
        deltaTokenQKV = matrixVectorMult ((wo (weighting network)) !! indexLayer) (reshapeMatrixToVector activations)

      shouldBeSmall 1.0 $ (V.sum deltaTokenQKV) - 2897446.0276938234
      shouldBeSmall 1.0 $ (V.maximum deltaTokenQKV) - 10245.32163638757
      shouldBeSmall 1.0 $ (V.minimum deltaTokenQKV) - 9862.115233500097

      Mx.nrows activations `shouldBe` 6
      Mx.ncols activations `shouldBe` 48
      shouldBeSmall 1e-1 $ (Mx.trace activations) - 407.2077331542969
      shouldBeSmall 1e-1 $ (sum (Mx.toList activations)) - 19385.04123687744
      shouldBeSmall 1e-1 $ (minimum (Mx.toList activations)) - 59.436580657958984
      shouldBeSmall 1e-1 $ (maximum (Mx.toList activations)) - 87.67498779296875

      length q `shouldBe` 6
      length k `shouldBe` 6
      length v `shouldBe` 6

      shouldBeSmall 1e-3 $ (k !! 0 V.! 0) - 13.5140090
      shouldBeSmall 1e-3 $ (k !! 0 V.! 47) - 76.510922895
      shouldBeSmall 1e-3 $ (k !! 1 V.! 0) - 4.048432575029892
      shouldBeSmall 1e-3 $ (k !! 1 V.! 47) - 75.99460779792548
      shouldBeSmall 1e-3 $ (k !! 2 V.! 0) - (-7.048659211879567)
      shouldBeSmall 1e-3 $ (k !! 2 V.! 47) - 74.24330840422249
      shouldBeSmall 1e-3 $ (k !! 3 V.! 0) - 11.798806806020366
      shouldBeSmall 1e-3 $ (k !! 3 V.! 47) - 71.89907238107276
      shouldBeSmall 1e-3 $ (k !! 4 V.! 0) - 12.957168370788168
      shouldBeSmall 1e-3 $ (k !! 4 V.! 47) - 74.44716220191003
      shouldBeSmall 1e-3 $ (k !! 5 V.! 0) - 10.358258170685986
      shouldBeSmall 1e-3 $ (k !! 5 V.! 47) - 74.42684423320338
 
      shouldBeSmall 1e-3 $ (q !! 0 V.! 0) - 10.047595305990399
      shouldBeSmall 1e-3 $ (q !! 0 V.! 47) - 73.5352357337324
      shouldBeSmall 1e-3 $ (q !! 1 V.! 0) - 14.202909536670631
      shouldBeSmall 1e-3 $ (q !! 1 V.! 47) - 67.21390186872122
      shouldBeSmall 1e-3 $ (q !! 2 V.! 0) - 5.207165646689646
      shouldBeSmall 1e-3 $ (q !! 2 V.! 47) - 76.75705530550135
      shouldBeSmall 1e-3 $ (q !! 3 V.! 0) - 9.709002411821984
      shouldBeSmall 1e-3 $ (q !! 3 V.! 47) - 93.7325735273318
      shouldBeSmall 1e-3 $ (q !! 4 V.! 0) - 5.700445560096341
      shouldBeSmall 1e-3 $ (q !! 4 V.! 47) - 73.207704757823
      shouldBeSmall 1e-3 $ (q !! 5 V.! 0) - 12.008699038487975
      shouldBeSmall 1e-3 $ (q !! 5 V.! 47) - 70.1946099367691

      shouldBeSmall 1e-3 $ (v !! 0 V.! 0) - 69.27178955078125
      shouldBeSmall 1e-3 $ (v !! 0 V.! 47) - 68.98504638671875
      shouldBeSmall 1e-3 $ (v !! 1 V.! 0) - 68.5321044921875
      shouldBeSmall 1e-3 $ (v !! 1 V.! 47) - 65.29499816894531
      shouldBeSmall 1e-3 $ (v !! 2 V.! 0) - 67.18028259277344
      shouldBeSmall 1e-3 $ (v !! 2 V.! 47) - 66.77857971191406
      shouldBeSmall 1e-3 $ (v !! 3 V.! 0) - 66.6872329711914
      shouldBeSmall 1e-3 $ (v !! 3 V.! 47) - 71.23432922363281
      shouldBeSmall 1e-3 $ (v !! 4 V.! 0) - 68.06254577636719
      shouldBeSmall 1e-3 $ (v !! 4 V.! 47) - 69.51446533203125
      shouldBeSmall 1e-3 $ (v !! 5 V.! 0) - 62.54541778564453
      shouldBeSmall 1e-3 $ (v !! 5 V.! 47) - 68.81326293945312

      length cacheKey' `shouldBe` 3
      length cacheValue' `shouldBe` 3
      shouldBeSmall 1e-3 $ (cacheKey' !! 2 !! 2 !! 0  V.! 0) - 13.514
      shouldBeSmall 1e-3 $ (cacheKey' !! 2 !! 2 !! 5  V.! 47) - 74.42679
      shouldBeSmall 1e-3 $ (cacheValue' !! 2 !! 2 !! 0  V.! 0) - 69.27178955
      shouldBeSmall 1e-3 $ (cacheValue' !! 2 !! 2 !! 5  V.! 47) - 68.813262939

      token V.! 0 `shouldBe` 0.616
      token V.! 287 `shouldBe` 0.176
      V.length token' `shouldBe` 288

      (V.take 5 token') `shouldBe` V.fromList [2439003.5, 2431244.5, 2431188.8, 2438949.8, 2426320.8]
      (V.take 5 (V.drop 283 token')) `shouldBe` V.fromList [2439129.5, 2428409.0, 2433409.0, 2427229.3, 2442822.5]

      token' V.! 0 `shouldBe` 2439003.5
      token' V.! 287 `shouldBe` 2442822.55
      V.sum token' `shouldBe` 7.0113574e8
      V.minimum token' `shouldBe` 2418157.3
      V.maximum token' `shouldBe` 2453980.0

vectorDistance :: (V.Vector Float) -> [Float] -> Float
vectorDistance vector array = V.sum (V.zipWith (-) vector (V.fromList array))

shouldBeSmall :: Float -> Float -> Expectation
shouldBeSmall threshold a = shouldSatisfy a (\x -> abs(x) < threshold)
