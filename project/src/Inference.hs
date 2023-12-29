{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Inference where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Binary.Get as BG
import qualified Data.Text.Encoding as TE
import qualified Data.Text as T
import qualified Data.Char as C
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified System.Random as R
import qualified Data.Vector.Unboxed as V
import qualified Data.Matrix.Unboxed as M

import Linear
import Control.Monad.State
import System.IO (hFlush, stdout)
import Control.Monad (replicateM, foldM)
import Data.Binary.Get (runGet, getInt32le, getFloatle)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Vector.Unboxed (Vector)
import Data.Matrix.Unboxed (Matrix)

import Debug.Trace

data RunCache = RunCache
    { keyCache :: [[[Vector Float]]]
    , valueCache :: [[[Vector Float]]]
    } deriving (Show)

data TransformerWeighting = TransformerWeighting
    { tokenEmbeddingTable :: Matrix Float
    , rmsAttWeight :: [Vector Float]
    , wq :: [Matrix Float]
    , wk :: [Matrix Float]
    , wv :: [Matrix Float]
    , wo :: [Matrix Float]
    , rmsFfnWeight :: [Vector Float]
    , w1 :: [Matrix Float]
    , w3 :: [Matrix Float]
    , w2 :: [Matrix Float]
    , rmsFinalWeight :: Vector Float
    , freqCisReal :: [Vector Float]
    , freqCisImag :: [Vector Float]
    } deriving (Show)

data Network = Network
    { dim :: Int
    , hiddenDim :: Int
    , nLayers :: Int
    , numAttentionHeads :: Int
    , numKeyValueHeads :: Int
    , vocabSize :: Int
    , seqLen :: Int
    , headDimension :: Int
    , weighting :: TransformerWeighting
    } deriving (Show)

readVector :: Int -> BG.Get (Vector Float)
readVector count = do
    values <- replicateM count getFloatle
    return $ V.fromList values

readVectors :: Int -> Int -> BG.Get [Vector Float]
readVectors nrows ncols = replicateM nrows (readVector ncols)

readMatrix :: Int -> Int -> BG.Get (Matrix Float)
readMatrix nrows ncols = do
    values <- replicateM (nrows * ncols) getFloatle
    return $ M.fromLists (DLS.chunksOf ncols values)

readMatrices :: Int -> Int -> Int -> BG.Get [Matrix Float]
readMatrices ndepth nrows ncols = replicateM ndepth (readMatrix nrows ncols)

initModel :: BSL.ByteString -> Network
initModel networkConfigFile = runGet (do
        dim <- getInt32le
        hiddenDim <- getInt32le
        nLayers <- getInt32le
        numAttentionHeads <- getInt32le
        numKeyValueHeads <- getInt32le
        vocabSize <- getInt32le
        seqLen <- getInt32le
        tokenEmbeddingTable <- readMatrix (fromIntegral vocabSize) (fromIntegral dim)
        rmsAttWeight <- readVectors (fromIntegral nLayers) (fromIntegral dim)
        wq <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        wk <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        wv <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        wo <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        rmsFfnWeight <- readVectors (fromIntegral nLayers) (fromIntegral dim)
        w1 <- readMatrices (fromIntegral nLayers) (fromIntegral hiddenDim) (fromIntegral dim)
        w2 <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral hiddenDim)
        w3 <- readMatrices (fromIntegral nLayers) (fromIntegral hiddenDim) (fromIntegral dim)
        rmsFinalWeight <- readVector (fromIntegral dim)
        freqCisReal <- readVectors (fromIntegral seqLen) (((fromIntegral dim) `div` (fromIntegral numAttentionHeads)) `div` 2)
        freqCisImag <- readVectors (fromIntegral seqLen) (((fromIntegral dim) `div` (fromIntegral numAttentionHeads)) `div` 2)

        let
            headDimension = dim `div` numAttentionHeads
            weighting = TransformerWeighting
              { tokenEmbeddingTable = tokenEmbeddingTable
              , rmsAttWeight = rmsAttWeight
              , wq = wq
              , wk = wk
              , wv = wv
              , wo = wo
              , rmsFfnWeight = rmsFfnWeight
              , w1 = w1
              , w2 = w2
              , w3 = w3
              , rmsFinalWeight = rmsFinalWeight
              , freqCisReal = freqCisReal
              , freqCisImag = freqCisImag
              }
        return $ Network
            { dim = fromIntegral dim
            , hiddenDim = fromIntegral hiddenDim
            , nLayers = fromIntegral nLayers
            , numAttentionHeads = fromIntegral numAttentionHeads
            , numKeyValueHeads = fromIntegral numKeyValueHeads
            , vocabSize = abs (fromIntegral vocabSize)
            , seqLen = fromIntegral seqLen
            , headDimension = fromIntegral headDimension
            , weighting = weighting
            }
        ) networkConfigFile

parseTokens :: BSL.ByteString -> Int -> ([T.Text], [Float])
parseTokens file size = (vocab, vocabScores)
  where
    readToken :: BG.Get (Float, T.Text)
    readToken = do
      score <- BG.getFloatle
      tokenSize <- BG.getInt32le
      bstr <- TE.decodeUtf8 . BSL.toStrict <$> BG.getLazyByteString (fromIntegral tokenSize)
      return (score, bstr)

    scoresAndStrings :: BG.Get [(Float, T.Text)]
    scoresAndStrings = replicateM size readToken

    vocabScores = fst <$> BG.runGet scoresAndStrings file
    vocab = snd <$> BG.runGet scoresAndStrings file

tokenizerInit :: BSL.ByteString -> Int -> ([T.Text], [Float])
tokenizerInit file size = parseTokens (BSL.drop 4 file) size

strLookup :: Text -> [T.Text] -> Int
strLookup occurrence = fromMaybe (-1) . DL.findIndex (occurrence ==)

processTokens :: [Int] -> [T.Text] -> [Float] -> [Int]
processTokens tokens vocab vocabScores = process tokens
  where
    process :: [Int] -> [Int]
    process tokens' =
      case findBestPair tokens' of
        Just (bestIdx, bestId) ->
          process (mergePair bestIdx bestId tokens')
        Nothing ->
          tokens'

    findBestPair :: [Int] -> Maybe (Int, Int)
    findBestPair tokens' = foldr checkPair Nothing (zip [0..] (zip tokens' (drop 1 tokens')))
      where
        checkPair :: (Int, (Int, Int)) -> Maybe (Int, Int) -> Maybe (Int, Int)
        checkPair (count, (tokenPrev, tokenNext)) acc =
          case strLookup ((vocab !! tokenPrev) `T.append` (vocab !! tokenNext)) vocab of
            pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, pos)
            _ -> acc

        bestScore :: Float
        bestScore = -1e10

    mergePair :: Int -> Int -> [Int] -> [Int]
    mergePair idx code tokens' =
      take idx tokens' ++ [code] ++ drop (idx + 2) tokens'

bpeEncode :: T.Text -> [T.Text] -> [Float] -> [Int]
bpeEncode prompt vocab vocabScores =
  let tokens = map (\char -> fromMaybe (error "Character not found in vocabulary") (DL.elemIndex (T.pack [char]) vocab)) (T.unpack prompt)
  in processTokens tokens vocab vocabScores

argmax :: (V.Unbox a, Ord a) => V.Vector a -> Int
argmax = V.maxIndex

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

computeQKV :: Network -> Int -> Vector Float -> Vector Float -> Vector Float -> ([Vector Float], [Vector Float], [Vector Float])
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
    fromVectors [buildActivation (headDimension network) indexLayer valueCache indexHead (scores indexHead)
                    | indexHead <- [0 .. numAttentionHeads network - 1]]
    where
      hd = headDimension network
      scores indexHead = V.toList $ softmax rawScores (V.length rawScores)
        where
          rawScores = computeScores hd keyCache indexLayer indexHead headsQ
      fromVectors :: [Vector Float] -> Matrix Float
      fromVectors vectorList = M.fromLists $ map V.toList vectorList

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
matrixVectorMult mat vec = V.fromList [ (V.sum . V.zipWith (*) vec) row | row <- M.toRows mat ]

splitVector :: Int -> V.Vector Float -> [V.Vector Float]
splitVector m vec = fmap V.fromList $ DLS.chunksOf ((V.length vec) `div` m) (V.toList vec)

dotProduct :: V.Vector Float -> V.Vector Float -> Float
dotProduct vec1 vec2 = V.sum $ elementsProduct vec1 vec2

elementsProduct:: V.Vector Float -> V.Vector Float -> V.Vector Float
elementsProduct vec1 vec2 = V.zipWith (*) vec1 vec2

vectorSum:: V.Vector Float -> V.Vector Float -> V.Vector Float
vectorSum vec1 vec2 = V.zipWith (+) vec1 vec2

reshapeMatrixToVector :: Matrix Float -> V.Vector Float
reshapeMatrixToVector = V.fromList . M.toList

rmsNorm :: Vector Float -> Vector Float -> Vector Float
rmsNorm vector weights =
  let ss = ((dotProduct vector vector) / fromIntegral (V.length vector)) + 1e-5
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
        deltaTokenQKV = matrixVectorMult (wO !! indexLayer) (reshapeMatrixToVector activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token'
        result = V.zipWith (+) token' deltaTokenFFN
    put (RunCache keyCache' valueCache')
    return result

transformer :: Int -> Int -> Network -> StateT RunCache IO (Vector Float)
transformer tokenCode stepCount network = do
    -- Getting the token embedding
    let token = M.takeRow (tokenEmbeddingTable (weighting network)) tokenCode

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
      then return (argmax logits)
    else do
      liftIO $ drawSample $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let tokenStr =
        if tokenCode == 1 && C.isSpace (T.head (vocab !! nextToken))
          then T.tail (vocab !! nextToken)
          else vocab !! nextToken
  return (tokenStr, nextToken)

generateTokens :: Network -> Int -> [Int] -> Float -> [Text] -> StateT RunCache IO [Text]
generateTokens network checkedMaxSteps promptTokens temperature vocab = go 0 []
  where
    go timestep result
      | timestep >= checkedMaxSteps = return result
      | otherwise = do
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature network vocab 1
        liftIO $ printf "%s " tokenStr
        liftIO $ hFlush stdout
        go (timestep + 1) (result ++ [tokenStr | nextToken /= 1])

run :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let
    seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
    network = initModel modelFileContent
    --weighting = checkpointInitWeights network modelFileContent
    (vocab, vocabScores) = tokenizerInit tokenizerFileContent (vocabSize network)
    promptTokens = bpeEncode (T.pack (fromMaybe "" prompt)) vocab vocabScores
    initCache = RunCache { keyCache = [], valueCache = [] }
  textList <- evalStateT (generateTokens network steps promptTokens temperature vocab) initCache
  printf "network: # layers %d / # attention heads %d / head dimension %d / vocabulary size %d\n" (nLayers network) (numAttentionHeads network) (headDimension network) (vocabSize network)
  printf "prompt tokens: %s\n" $ show promptTokens
  printf "initial sentence: %s\n" $ show $ map (\token -> vocab !! token) promptTokens
  printf "seed value %d, temperature %f\n" seedValue temperature
  putStrLn $ T.unpack $ T.intercalate (T.pack " ") textList
