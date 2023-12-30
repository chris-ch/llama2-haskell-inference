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

softmax :: Vector Float -> Int -> Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (\x -> x / sumExpValues) expValues

drawSample :: Vector Float -> IO Int
drawSample probabilities = do
  r <- R.randomIO :: IO Float
  let cdf = DL.scanl1 (+) (V.toList probabilities)
  return $ go cdf r 0
  where
    go (p:ps) r acc
      | r < p = acc
      | otherwise = go ps r (acc + 1)
    go _ _ acc = acc

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

replaceAtIndex :: Int -> a -> [a] -> [a]
replaceAtIndex index newValue list
  | index < 0 || index >= length list = list
  | otherwise = take index list ++ [newValue] ++ drop (index + 1) list

createLayerToken :: Network -> Int -> Int -> Vector Float -> Vector Float -> Vector Float -> StateT RunCache IO (Vector Float)
createLayerToken network stepCount indexLayer freqCisRealRow freqCisImagRow token = do
    (keyCache, valueCache) <- gets (\cache -> (keyCache cache, valueCache cache))
    let
        (headsQ, headsK, headsV) = computeQKV network indexLayer freqCisRealRow freqCisImagRow token
        keyCacheStep = (keyCache !! stepCount) ++ [headsK]
        valueCacheStep = (valueCache !! stepCount) ++ [headsV]
        keyCache' = replaceAtIndex stepCount keyCacheStep keyCache
        valueCache' = replaceAtIndex stepCount valueCacheStep valueCache
        activations = multiheadActivation network indexLayer keyCache' valueCache' headsQ
        wO = wo (weighting network)
        deltaTokenQKV = matrixVectorMult (wO !! indexLayer) (V.concat activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token'
        result = V.zipWith (+) token' deltaTokenFFN
    put (RunCache keyCache' valueCache')
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

generateNextToken :: Int -> [Int] -> Float -> Network -> [Text] -> Int -> StateT RunCache IO (Text, Int)
generateNextToken timestep promptTokens temperature network vocab tokenCode = do
  logits <- transformer tokenCode timestep network
  nextToken <- if timestep < length promptTokens
    then return (promptTokens !! timestep)
    else if temperature == 0.0
      then return (V.maxIndex logits)
    else do
      liftIO $ drawSample $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let tokenStr =
        if tokenCode == 1 && C.isSpace (T.head (vocab !! nextToken))
          then T.tail (vocab !! nextToken)
          else vocab !! nextToken
  return (tokenStr, nextToken)

generateTokens :: Network -> Int -> [Int] -> Float -> [Text] -> StateT RunCache IO [Text]
generateTokens network checkedMaxSteps promptTokens temperature vocab = go 0 [] 1
  where
    go timestep result token
      | timestep >= checkedMaxSteps = return result
      | otherwise = do
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature network vocab token
        liftIO $ printf "%s" tokenStr
        liftIO $ hFlush stdout
        go (timestep + 1) (result ++ [tokenStr | nextToken /= 1]) nextToken

run :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let
    seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
    network = initModel modelFileContent
    --weighting = checkpointInitWeights network modelFileContent
    (vocab, vocabScores) = tokenizerInit tokenizerFileContent (vocabSize network)
    promptTokens = bpeEncode (T.pack (fromMaybe "" prompt)) vocab vocabScores
    initCache :: RunCache
    initCache = RunCache { keyCache = [], valueCache = [] }
  textList <- evalStateT (generateTokens network steps promptTokens temperature vocab) initCache
  printf "network: # layers %d / # attention heads %d / head dimension %d / vocabulary size %d\n" (nLayers network) (numAttentionHeads network) (headDimension network) (vocabSize network)
  printf "prompt tokens: %s\n" $ show promptTokens
  printf "initial sentence: %s\n" $ show $ map (\token -> vocab !! token) promptTokens
  printf "seed value %d, temperature %f\n" seedValue temperature
  putStrLn $ T.unpack $ T.intercalate (T.pack " ") textList
