{-# LANGUAGE FlexibleContexts #-}

module Inference (run, computeQKV, rmsNorm,
computeDeltaFFN, createLayerToken, multiheadActivation,
buildActivation, applyRotations, transformer,
softmax, drawSample
 ) where

import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified System.Random as R
import qualified Data.Vector.Unboxed as V
import qualified Matrix as M

import NetworkBuilder (NetworkConfig(..), AttentionKV(..),
  TransformerWeighting(..), KeyCache, ValueCache, Vocabulary, PromptTokens, Token,
  initModel, tokenizerInit)
import Control.Monad.State ( StateT, evalStateT, MonadState(put), gets )
import System.IO (hFlush, stdout)
import Control.Monad (foldM)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Vector.Unboxed (Vector)
import Data.Time.Clock.POSIX (getPOSIXTime)
import Control.Monad.Reader
    ( MonadIO(liftIO),
      ReaderT(runReaderT),
      MonadReader(ask) )
import GHC.IO.Handle (Handle)
import GHC.Unicode (isSpace)

type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

softmax :: Vector Float -> Int -> Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (/ sumExpValues) expValues

drawSample :: Int -> Vector Float -> IO Token
drawSample seedValue probabilities = do
  let
    gen = R.mkStdGen seedValue
    (r, _) = R.random gen :: (Float, R.StdGen)

    indexHighestCDF :: Float -> Vector Float -> Int
    indexHighestCDF rand vec = min (V.ifoldl' (indexHighest rand) 0 cdf) (V.length vec - 1)
        where
          cdf = V.scanl1 (+) vec
          indexHighest :: Float -> Int -> Int -> Float -> Int
          indexHighest rand' acc i v = if v <= rand' then i + 1 else acc

  return $ fromIntegral $ indexHighestCDF r probabilities

-- Returns an array of size (headDim)
buildActivation :: Int -> Int -> ValueCache -> Int -> Vector Float -> Vector Float
buildActivation headDim indexLayer vC indexHead headScores =
  DL.foldl' accumulate zeroVector zippedValues
  where
    accumulate :: Vector Float -> (Vector Float, Float) -> Vector Float
    accumulate acc (valueVector, attentionWeight) = V.zipWith (+) acc (scale attentionWeight valueVector)
    zeroVector = V.replicate headDim 0.0
    zippedValues = zip (map (\countToken -> M.getRowVector (vC !! countToken !! indexLayer) indexHead) [0..]) (V.toList headScores)
    scale w = V.map (w *)

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

dotProduct :: Vector Float -> Vector Float -> Float
dotProduct vec1 vec2 = V.sum $ V.zipWith (*) vec1 vec2

rmsNorm :: Vector Float -> Vector Float -> Vector Float
rmsNorm vector weights =
  let
    squareNorm :: Vector Float -> Float
    squareNorm = V.foldl' cumSumSquare 0.0
      where
        cumSumSquare :: Float -> Float -> Float
        cumSumSquare acc v = acc + v ^ (2::Int)

    ss = (squareNorm vector / fromIntegral (V.length vector)) + 1e-5
    normalized = V.map (* (1.0 / sqrt ss)) vector
  in V.zipWith (*) weights normalized

computeDeltaFFN :: TransformerWeighting -> Int -> Vector Float -> Vector Float
computeDeltaFFN weights indexLayer token =
    let
      sigmoidLinearUnit :: Float -> Float
      sigmoidLinearUnit value = value / (1.0 + exp (-value))

      rmsFFNWeight = rmsFfnWeight weights !! indexLayer
      weight1 = w1 weights !! indexLayer
      weight2 = w2 weights !! indexLayer
      weight3 = w3 weights !! indexLayer
      rba = rmsNorm token rmsFFNWeight
      hiddenDimensionBuffer1 = M.multiplyVector weight1 rba
      hiddenDimensionBuffer2 = M.multiplyVector weight3 rba
      sigmoided = V.map sigmoidLinearUnit hiddenDimensionBuffer1
    in M.multiplyVector weight2 (V.zipWith (*) sigmoided hiddenDimensionBuffer2)

computeQKV :: TransformerWeighting -> Int -> Int -> Int -> Vector Float -> Vector Float -> Vector Float -> (M.Matrix Float, M.Matrix Float, M.Matrix Float)
computeQKV weights numHeads dimHead indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token (rmsAttWeight weights !! indexLayer)
    wQ = M.Matrix numHeads dimHead $ M.multiplyVector (wq weights !! indexLayer) rba
    headsQ = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) (M.getRowVectors wQ)
    wK = M.Matrix numHeads dimHead $ M.multiplyVector (wk weights !! indexLayer) rba
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) (M.getRowVectors wK)
    headsV = M.multiplyVector (wv weights !! indexLayer) rba
  in
    (M.fromVectors numHeads dimHead headsQ, M.fromVectors numHeads dimHead headsK, M.Matrix numHeads dimHead headsV)

computeScores :: Int -> KeyCache -> Int -> Int -> M.Matrix Float -> Vector Float
computeScores headDim kC indexLayer indexHead headsQ = V.fromList $ map calculateScore kC
  where
    calculateScore :: [M.Matrix Float] -> Float
    calculateScore keyVectors =
      let keyVector = M.getRowVector (keyVectors !! indexLayer) indexHead
      in dotProduct (M.getRowVector headsQ indexHead) keyVector / sqrt (fromIntegral headDim)

multiheadActivation :: Int -> Int -> Int -> KeyCache -> ValueCache -> M.Matrix Float -> M.Matrix Float
multiheadActivation numHeads headDim indexLayer kC vC headsQ =
    M.fromVectors numHeads headDim [activation indexHead | indexHead <- [0 .. numHeads - 1]]
    where
      activation :: Int -> Vector Float
      activation indexHead = buildActivation headDim indexLayer vC indexHead (scores indexHead)
      scores indexHead = softmax rawScores (V.length rawScores)
        where
          rawScores = computeScores headDim kC indexLayer indexHead headsQ

createLayerToken :: Int -> Int -> Vector Float -> Vector Float -> Vector Float -> TransformerResult (Vector Float)
createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow token = do
    network <- ask
    (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
    let
        (headsQ, headsK, headsV) = computeQKV (weighting network) (numAttentionHeads network) (headDimension network) indexLayer freqCisRealRow freqCisImagRow token
        keyCacheStep = (kC !! stepCount) ++ [headsK]
        valueCacheStep = (vC !! stepCount) ++ [headsV]
        keyCache' = take stepCount kC ++ [keyCacheStep]
        valueCache' = take stepCount vC ++ [valueCacheStep]
        activations = multiheadActivation (numAttentionHeads network) (headDimension network) indexLayer keyCache' valueCache' headsQ
        wO = wo (weighting network)
        deltaTokenQKV = M.multiplyVector (wO !! indexLayer) (M.values activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token'
        result = V.zipWith (+) token' deltaTokenFFN
    put (AttentionKV {keyCache = keyCache', valueCache = valueCache'})
    return result

transformer :: Token -> Int -> TransformerResult (Vector Float)
transformer tokenCode stepCount = do
    network <- ask

    -- Getting the token embedding
    let token = M.getRowVector (tokenEmbeddingTable (weighting network)) (fromIntegral tokenCode)

    -- Plucking out the current row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = freqCisReal (weighting network) !! stepCount
    let freqCisImagRow = freqCisImag (weighting network) !! stepCount

    -- Forwarding all the layers
    finalToken <- foldM (\accToken indexLayer -> createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow accToken)
                  token
                  [0..numLayers network - 1]

    -- Final rmsnorm
    let tokenWithRms = rmsNorm finalToken (rmsFinalWeight $ weighting network)

    -- Classifier into logits
    let logits = M.multiplyVector (tokenEmbeddingTable (weighting network)) tokenWithRms

    return logits

generateNextToken :: Int -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> TransformerResult (BS.ByteString, Token)
generateNextToken timestep promptTokens temperature vocab tokenCode seedValue = do
  network <- ask
  logits <- transformer tokenCode timestep
  nextToken <- if timestep < length promptTokens
    then return (promptTokens !! timestep)
    else if temperature == 0.0
      then return $ fromIntegral (V.maxIndex logits)
    else do
      liftIO $ drawSample seedValue $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let
    word = vocab !! fromIntegral nextToken :: BS.ByteString
    firstChar = BSC.head word :: Char
    tokenStr = if tokenCode == 1 && isSpace firstChar
          then BSC.tail (vocab !! fromIntegral nextToken)
          else vocab !! fromIntegral nextToken
  return (tokenStr, nextToken)

generateTokens :: Int -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerResult ([BS.ByteString], Int)
generateTokens maxSteps promptTokens temperature vocab seedValue = do
  network <- ask
  go network 0 [] 1 where
    go network timestep result token
      | timestep >= maxSteps || (timestep /= 0 && token == 1) = return (result, timestep)
      | otherwise = do
        (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
        put (AttentionKV {keyCache = take timestep kC ++ [[]], valueCache = take timestep vC ++ [[]]})
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature vocab token seedValue
        liftIO $ printf "%s" (BSC.unpack tokenStr)
        liftIO $ hFlush stdout
        go network (timestep + 1) (result ++ [tokenStr]) nextToken

run :: Handle -> Handle -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileHandle tokenizerFileHandle temperature steps prompt seed = do
  currentTime <- getPOSIXTime

  modelFileContent <- BS.hGetContents modelFileHandle
  tokenizerFileContent <- BS.hGetContents tokenizerFileHandle
  putStrLn "running inference..."

  let
    seedValue = fromMaybe (round currentTime) seed
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')
    initStateAttentionKV :: AttentionKV
    initStateAttentionKV = AttentionKV { keyCache = [], valueCache = [] }

  printf "network: # layers %d\n" (numLayers config)
  printf "network: # attention heads %d / head dimension %d / kv heads %d\n" (numAttentionHeads config) (headDimension config) (numKeyValueHeads config)
  printf "network: vocabulary size %d\n" $ vocabSize config
  printf "network: token dimensions %d\n" $ tokenDim config
  printf "prompt tokens: %s\n" $ show promptTokens
  printf "seed value %d, temperature %f\n" seedValue temperature

  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, countTokens) <- evalStateT (runReaderT (generateTokens steps promptTokens temperature vocab seedValue) config) initStateAttentionKV
  endTime <- getPOSIXTime
  let
    duration :: Integer
    duration = round (endTime - startTime)
    tokensPerSec :: Float
    tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "duration: %ds - (%.02f tokens/s)\n" duration tokensPerSec

  return ()
