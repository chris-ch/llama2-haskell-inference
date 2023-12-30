{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Inference where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Text as T
import qualified Data.Char as C
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified System.Random as R
import qualified Data.Vector.Unboxed as V

import Builder (Network(..), RunCache(..),
  Matrix, TransformerWeighting(..),
  initModel, tokenizerInit, bpeEncode)
import Control.Monad.State
import System.IO (hFlush, stdout)
import Control.Monad (foldM)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Vector.Unboxed (Vector)
import Data.Time.Clock.POSIX (getPOSIXTime)

softmax :: Vector Float -> Int -> Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (\x -> x / sumExpValues) expValues

indexHighestCDF :: Float -> Vector Float -> Int
indexHighestCDF r vec = min (V.ifoldl' (indexHighest r) 0 cdf) (V.length vec - 1)
    where
      cdf = V.scanl1 (+) vec
      indexHighest :: Float -> Int -> Int -> Float -> Int
      indexHighest rand acc i v = if v <= rand then i + 1 else acc

drawSample :: Int -> Vector Float -> IO Int
drawSample seedValue probabilities = do
  let gen = R.mkStdGen seedValue
  let (r, _) = R.random gen :: (Float, R.StdGen)
  return $ indexHighestCDF r probabilities

computeQKV :: Builder.Network -> Int -> Vector Float -> Vector Float -> Vector Float -> ([Vector Float], [Vector Float], [Vector Float])
computeQKV network indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)
    wQ = splitVector (numAttentionHeads network) (matrixVectorMult ((wq (weighting network)) !! indexLayer) rba)
    headsQ = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wQ
    wK = splitVector (numAttentionHeads network) (matrixVectorMult ((wk (weighting network)) !! indexLayer) rba)
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wK
    headsV = splitVector (numAttentionHeads network) (matrixVectorMult ((wv (weighting network)) !! indexLayer) rba)
  in
    (headsQ, headsK, headsV)

multiheadActivation :: Network -> Int -> [[[Vector Float]]]-> [[[Vector Float]]] -> [Vector Float] -> Matrix Float
multiheadActivation network indexLayer keyCache valueCache headsQ = 
    [buildActivation (headDimension network) indexLayer valueCache indexHead (scores indexHead)
                    | indexHead <- [0 .. numAttentionHeads network - 1]]
    where
      hd = headDimension network
      scores indexHead = V.toList $ softmax rawScores (V.length rawScores)
        where
          rawScores = computeScores hd keyCache indexLayer indexHead headsQ

buildActivation :: Int -> Int -> [[[Vector Float]]] -> Int -> [Float] -> Vector Float
buildActivation dimension indexLayer valueCache indexHead headScores =
  DL.foldl' accumulate zeroVector zippedValues
  where
    accumulate :: (Vector Float) -> ((Vector Float), Float) -> (Vector Float)
    accumulate acc (valueVector, attentionWeight) = V.zipWith (+) acc (scale attentionWeight valueVector)
    scale w vec = V.map (\x -> w * x) vec
    zeroVector = V.replicate dimension 0.0
    zippedValues = zip (map (\count -> valueCache !! count !! indexLayer !! indexHead) [0..]) headScores

computeScores :: Int -> [[[Vector Float]]] -> Int -> Int -> [Vector Float] -> Vector Float
computeScores headDimension keyCache indexLayer indexHead headsQ = V.fromList $ map calculateScore keyCache
  where
    calculateScore :: [[Vector Float]] -> Float
    calculateScore keyVectors = 
      let keyVector = ((keyVectors !! indexLayer) !! indexHead) 
          score = (dotProduct (headsQ !! indexHead) keyVector) / sqrt (fromIntegral (headDimension))
      in score

applyRotations :: Vector Float -> Vector Float -> Vector Float -> Vector Float
applyRotations headVector freqCisRealRow freqCisImagRow =
  V.fromList $ concatMap applyRotation [0,2..V.length headVector - 2]
  where
    applyRotation :: Int -> [Float]
    applyRotation headItemIndex = [value * real - valueNext * imag, value * imag + valueNext * real]
      where
        real = freqCisRealRow V.! (headItemIndex `div` 2)
        imag = freqCisImagRow V.! (headItemIndex `div` 2)
        value = headVector V.! headItemIndex
        valueNext = headVector V.! (headItemIndex + 1)

matrixVectorMult :: Matrix Float -> Vector Float -> Vector Float
--matrixVectorMult mat vec = V.fromList [ (V.sum . V.zipWith (*) vec) row | row <- mat ]
matrixVectorMult mat vec = V.fromList $ map (\v -> dotProduct v vec) mat

splitVector :: Int -> Vector Float -> [Vector Float]
splitVector m vec = fmap V.fromList $ DLS.chunksOf ((V.length vec) `div` m) (V.toList vec)

dotProduct :: Vector Float -> Vector Float -> Float
dotProduct vec1 vec2 = V.sum $ elementsProduct vec1 vec2

elementsProduct :: Vector Float -> Vector Float -> Vector Float
elementsProduct vec1 vec2 = V.zipWith (*) vec1 vec2

squareNorm :: Vector Float -> Float
squareNorm vec = V.foldl sumSquare 0.0 vec
  where
    sumSquare :: Float -> Float -> Float
    sumSquare acc v = acc + (v ^ (2::Int))

rmsNorm :: Vector Float -> Vector Float -> Vector Float
rmsNorm vector weights =
  let ss = ((squareNorm vector) / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
  in elementsProduct weights normalized

sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit value = value / (1.0 + exp (-value))

computeDeltaFFN :: TransformerWeighting -> Int -> Vector Float -> Vector Float
computeDeltaFFN weighting indexLayer token =
    let
      rmsFFNWeight = (rmsFfnWeight weighting) !! indexLayer
      weight1 = (w1 weighting) !! indexLayer
      weight2 = (w2 weighting) !! indexLayer
      weight3 = (w3 weighting) !! indexLayer
      rba = rmsNorm token rmsFFNWeight
      hiddenDimensionBuffer1 = matrixVectorMult weight1 rba
      hiddenDimensionBuffer2 = matrixVectorMult weight3 rba
      sigmoided = V.map sigmoidLinearUnit hiddenDimensionBuffer1
    in matrixVectorMult weight2 (elementsProduct sigmoided hiddenDimensionBuffer2)

createLayerToken :: Network -> Int -> Int -> Vector Float -> Vector Float -> Vector Float -> StateT RunCache IO (Vector Float)
createLayerToken network stepCount indexLayer freqCisRealRow freqCisImagRow token = do
    (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
    let
        (headsQ, headsK, headsV) = computeQKV network indexLayer freqCisRealRow freqCisImagRow token
        keyCacheStep = (kC !! stepCount) ++ [headsK]
        valueCacheStep = (vC !! stepCount) ++ [headsV]
        keyCache' = take stepCount kC ++ [keyCacheStep]
        valueCache' = take stepCount vC ++ [valueCacheStep]
        activations = multiheadActivation network indexLayer keyCache' valueCache' headsQ
        wO = wo (weighting network)
        deltaTokenQKV = matrixVectorMult (wO !! indexLayer) (V.concat activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token'
        result = V.zipWith (+) token' deltaTokenFFN
    put (RunCache {keyCache = keyCache', valueCache = valueCache'})
    return result

transformer :: Int -> Int -> Network -> StateT RunCache IO (Vector Float)
transformer tokenCode stepCount network = do
    -- Getting the token embedding
    let token = (tokenEmbeddingTable (weighting network)) !! tokenCode

    -- Plucking out the current row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = freqCisReal (weighting network) !! stepCount
    let freqCisImagRow = freqCisImag (weighting network) !! stepCount

    -- Forwarding all the layers
    finalToken <- foldM (\accToken indexLayer -> createLayerToken network stepCount indexLayer freqCisRealRow freqCisImagRow accToken)
                  token
                  [0..nLayers network - 1]

    -- Final rmsnorm
    let tokenWithRms = rmsNorm finalToken (rmsFinalWeight $ weighting network)

    -- Classifier into logits
    let logits = matrixVectorMult (tokenEmbeddingTable (weighting network)) tokenWithRms

    return logits

generateNextToken :: Int -> [Int] -> Float -> Network -> [Text] -> Int -> Int -> StateT RunCache IO (Text, Int)
generateNextToken timestep promptTokens temperature network vocab tokenCode seedValue = do
  logits <- transformer tokenCode timestep network
  nextToken <- if timestep < length promptTokens
    then return (promptTokens !! timestep)
    else if temperature == 0.0
      then return (V.maxIndex logits)
    else do
      liftIO $ drawSample seedValue $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let tokenStr =
        if tokenCode == 1 && C.isSpace (T.head (vocab !! nextToken))
          then T.tail (vocab !! nextToken)
          else vocab !! nextToken
  return (tokenStr, nextToken)

generateTokens :: Network -> Int -> [Int] -> Float -> [Text] -> Int -> StateT RunCache IO ([Text], Int)
generateTokens network checkedMaxSteps promptTokens temperature vocab seedValue = go 0 [] 1
  where
    go timestep result token
      | timestep >= checkedMaxSteps || (timestep /= 0 && token == 1) = return (result, timestep)
      | otherwise = do
        (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
        put (RunCache {keyCache = take timestep kC ++ [[]], valueCache = take timestep vC ++ [[]]})
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature network vocab token seedValue
        liftIO $ printf "%s" tokenStr
        liftIO $ hFlush stdout
        go (timestep + 1) (result ++ [tokenStr]) nextToken

run :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromMaybe (round currentTime) seed
    network = initModel modelFileContent
    (vocab, vocabScores) = tokenizerInit tokenizerFileContent (vocabSize network)
    promptTokens = bpeEncode (T.pack (fromMaybe "" prompt)) vocab vocabScores
    initCache :: RunCache
    initCache = RunCache { keyCache = [], valueCache = [] }
  printf "network: # layers %d / # attention heads %d / head dimension %d / vocabulary size %d\n" (nLayers network) (numAttentionHeads network) (headDimension network) (vocabSize network)
  printf "prompt tokens: %s\n" $ show promptTokens
  printf "initial sentence: %s\n" $ show $ map (\token -> vocab !! token) promptTokens
  printf "seed value %d, temperature %f\n" seedValue temperature
  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, countTokens) <- evalStateT (generateTokens network steps promptTokens temperature vocab seedValue) initCache
  endTime <- getPOSIXTime
  let
    duration :: Integer
    duration = round (endTime - startTime)
    tokensPerSec :: Float
    tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "duration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
