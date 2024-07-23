{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module TokenGeneratorSpec (spec) where

import Test.Hspec ( Spec, describe, it, shouldBe )
import NetworkBuilder
    ( NetworkConfig(..),
      AttentionKV(SAttentionKV, sValueCache, sKeyCache) )
import Inference ( softmax, drawSample, transformer )
import CustomRandom
    ( CustomRNG, generateRandomVector, buildRandomNetworkConfig )

import qualified Data.Vector.Unboxed as V
import Control.Monad.State ( evalState )
import Control.Monad ( replicateM, forM_ )
import Control.Monad.Reader ( ReaderT(runReaderT), MonadTrans (lift) )
import Control.Monad.ST.Trans (runSTT)
import qualified Data.Array.ST as AST

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

      logits <- runSTT $ runReaderT (do
          let 
            numHeads = numAttentionHeads network
            numHeadComponents = headDimension network
            
          sKC <- lift $ AST.newArray ((0, 0, 0, 0), (nSteps - 1, numLayers - 1, numHeads - 1, numHeadComponents - 1)) 0.0
          sVC <- lift $ AST.newArray ((0, 0, 0, 0), (nSteps - 1, numLayers - 1, numHeads - 1, numHeadComponents - 1)) 0.0

          -- Initialize sKC with values from cacheKey
          forM_ [0..2] $ \i -> do
              let vector2 = cacheKey !! i
              forM_ [0..(length vector2) - 1] $ \j -> do
                  let vector1 = cacheKey !! i !! j
                  forM_ [0..(length vector1) - 1] $ \k -> do
                      let vector0 = cacheKey !! i !! j !! k
                      forM_ [0..(V.length vector0) - 1] $ \l -> do
                          lift $ AST.writeArray sKC (i, j, k, l) (vector0 V.! l)

          -- Initialize sVC with values from cacheValue
          forM_ [0..2] $ \i -> do
              let vector2 = cacheKey !! i
              forM_ [0..(length vector2) - 1] $ \j -> do
                  let vector1 = cacheKey !! i !! j
                  forM_ [0..(length vector1) - 1] $ \k -> do
                      let vector0 = cacheValue !! i !! j !! k
                      forM_ [0..(V.length vector0) - 1] $ \l -> do
                          lift $ AST.writeArray sVC (i, j, k, l) (vector0 V.! l)


          let sakv = SAttentionKV { sKeyCache = sKC, sValueCache = sVC }
          transformer stepCount tokenCode sakv
        ) network

      (V.take 5 logits) `shouldBe` V.fromList [76.48803,75.86596,69.82339,73.1698,72.237274]
      (V.take 5 (V.drop 31995 logits)) `shouldBe` V.fromList [81.340195,77.3984,78.24091,82.833206,75.389984]

      next_token <- drawSample 11284652 $ softmax (V.map (/ 0.8) logits) (vocabSize network)
      next_token `shouldBe` 27569
