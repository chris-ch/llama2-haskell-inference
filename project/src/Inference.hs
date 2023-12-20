{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Inference( run ) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Binary.Get as BG
import qualified Data.Text.Encoding as TE
import qualified Data.Text as T
import qualified Data.Char as C
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified System.Random as R
import qualified Data.Vector as V
import qualified Data.Matrix as M

import Control.Monad (replicateM)
import Data.Binary.Get (runGet, getInt32le, getFloatle)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Vector (Vector)
import Data.Matrix (Matrix)

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
    , freqCisImag ::[Vector Float]
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

type TokenCode = Int
type Timestep = Int
type MaxSteps = Int
type Temperature = Float
type Tokens = [Int]
type Vocab = [Text]

readVector :: Int -> BG.Get (Vector Float)
readVector count = do
    values <- replicateM (count) getFloatle
    return $ V.fromList values

readMatrix :: Int -> Int -> BG.Get (Matrix Float)
readMatrix nrows ncols = do
    values <- replicateM (nrows * ncols) getFloatle
    return $ M.fromLists (DLS.chunksOf ncols values)

readMatrices :: Int -> Int -> Int -> BG.Get [Matrix Float]
readMatrices ndepth nrows ncols = do
  values <- replicateM (nrows * ncols * ndepth) getFloatle
  let
    chunkSize = nrows * ncols
    matrices = [ M.fromList nrows ncols $ take chunkSize (drop (i * chunkSize) values) | i <- [0 .. (length values `div` chunkSize) - 1]]
  return matrices

readVectors :: Int -> Int -> BG.Get [Vector Float]
readVectors nrows ncols = do
  values <- replicateM (nrows * ncols) getFloatle
  return [ V.fromList $ take ncols (drop (i * ncols) values) | i <- [0 .. (length values `div` ncols) - 1]]

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
        w1 <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        w2 <- readMatrices (fromIntegral nLayers) (fromIntegral hiddenDim) (fromIntegral dim)
        w3 <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral hiddenDim)
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

generateTokens :: Network -> RunCache -> Int -> [Int] -> Float -> [Text] -> IO [Text]
generateTokens network cache checkedMaxSteps promptTokens temperature vocab =
  go 0 []
  where
    go timestep result
      | timestep >= checkedMaxSteps = return result
      | otherwise = do
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature network vocab 1 cache
        go (timestep + 1) (result ++ [tokenStr | nextToken /= 1])

argmax :: (Ord a) => V.Vector a -> Int
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

transformer :: Int -> Int -> Network -> RunCache -> Vector Float
transformer tokenCode timestep network cache = V.replicate (vocabSize network) 1.0  -- Replace this with your actual implementation

computeQKV :: Network -> Int -> Vector Float -> Vector Float -> Vector Float -> ([Vector Float], [Vector Float], [Vector Float])
computeQKV network indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)
    wQ = splitVector (numAttentionHeads network) (matrixVectorMult ((wq (weighting network)) !! indexLayer) rba)
    headsQ = map (\vector -> applyRotations network vector freqCisRealRow freqCisImagRow) wQ
    wK = splitVector (numAttentionHeads network) (matrixVectorMult ((wk (weighting network)) !! indexLayer) rba)
    headsK = map (\vector -> applyRotations network vector freqCisRealRow freqCisImagRow) wK

    headsV = splitVector (numAttentionHeads network) (matrixVectorMult ((wv (weighting network)) !! indexLayer) rba)
  in
    (headsQ, headsK, headsV)

multiheadActivation :: Network -> Int -> [[[Vector Float]]]-> [[[Vector Float]]] -> [Vector Float] -> Matrix Float
multiheadActivation network indexLayer keyCache valueCache headsQ = 
    fromVectors [buildActivation hd indexLayer valueCache indexHead (scores indexHead)
                    | indexHead <- [0 .. numAttentionHeads network - 1]]
    where
      hd = headDimension network
      scores indexHead = V.toList $ softmax rawScores (V.length rawScores)
        where
          rawScores = computeScores hd keyCache indexLayer indexHead headsQ
      fromVectors :: [Vector Float] -> Matrix Float
      fromVectors vectorList = M.fromLists $ map V.toList vectorList

buildActivation :: Int -> Int -> [[[Vector Float]]] -> Int -> [Float] -> Vector Float
buildActivation dim indexLayer valueCache indexHead headScores =
  let numHeads = length valueCache
      valueVectors = [ valueCache !! i !! indexLayer !! indexHead | i <- [0 .. numHeads - 1]]
      multiplyWithAttention i = V.map (* headScores !! i) (valueVectors !! i)
      activations = [multiplyWithAttention i | i <- [0 .. numHeads - 1]]
  in  foldl vectorSum (V.replicate numHeads 0.0) activations

computeScores :: Int -> [[[Vector Float]]] -> Int -> Int -> [Vector Float] -> Vector Float
computeScores headDimension keyCache indexLayer indexHead headsQ = V.fromList $ map calculateScore keyCache
  where
    calculateScore :: [[Vector Float]] -> Float
    calculateScore keyVectors = 
      let keyVector = ((keyVectors !! indexLayer) !! indexHead) 
          score = (dotProduct (headsQ !! indexHead) keyVector) / sqrt (fromIntegral (headDimension))
      in score

applyRotations :: Network -> Vector Float -> Vector Float -> Vector Float -> Vector Float
applyRotations network head freqCisRealRow freqCisImagRow =
    V.generate (V.length head) handleItem
  where
    handleItem i
      | even i = 
          let real = freqCisRealRow V.! (i `div` 2)
              imag = freqCisImagRow V.! (i `div` 2)
              value = head V.! i
              valueNext = head V.! (i + 1)
          in value * real - valueNext * imag
      | otherwise = 
          let real = freqCisRealRow V.! (i `div` 2)
              imag = freqCisImagRow V.! (i `div` 2)
              value = head V.! i
              valueNext = head V.! (i - 1)
          in value * imag + valueNext * real

generateNextToken :: Int -> [Int] -> Float -> Network -> [Text] -> Int -> RunCache -> IO (Text, Int)
generateNextToken timestep promptTokens temperature network vocab tokenCode cache = do
  let logits = transformer tokenCode timestep network cache
  nextToken <- if timestep < length promptTokens
    then return (promptTokens !! timestep)
    else if temperature == 0.0
      then return (argmax logits)
      else drawSample $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let tokenStr =
        if tokenCode == 1 && C.isSpace (T.head (vocab !! nextToken))
          then T.tail (vocab !! nextToken)
          else vocab !! nextToken
  return (tokenStr, nextToken)

matrixVectorMult :: (Num a) => Matrix a -> Vector a -> Vector a
matrixVectorMult matrix vector = M.getCol 0 $ M.multStd2 matrix (M.colVector vector)

splitVector :: (Num a) => Int -> V.Vector a -> [V.Vector a]
splitVector m vec = V.toList $ V.concatMap (V.singleton . V.slice 0 m) $ V.iterateN m V.tail vec

dotProduct :: (Num a) => V.Vector a -> V.Vector a -> a
dotProduct vec1 vec2 = V.sum $ elementsProduct vec1 vec2

elementsProduct:: (Num a) => V.Vector a -> V.Vector a -> V.Vector a
elementsProduct vec1 vec2 = V.zipWith (*) vec1 vec2

vectorSum:: (Num a) => V.Vector a -> V.Vector a -> V.Vector a
vectorSum vec1 vec2 = V.zipWith (+) vec1 vec2

reshapeMatrixToVector :: Matrix a -> V.Vector a
reshapeMatrixToVector = V.fromList . M.toList

rmsNorm :: Vector Float -> Vector Float -> Vector Float
rmsNorm vector weights =
  let ss = ((dotProduct vector vector) / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
  in  elementsProduct weights normalized

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

createLayerToken :: Network -> Int -> [[[Vector Float]]] -> [[[Vector Float]]] -> Int -> Vector Float -> Vector Float -> Vector Float -> (Vector Float, [[[Vector Float]]], [[[Vector Float]]])
createLayerToken network stepCount keyCache valueCache indexLayer freqCisRealRow freqCisImagRow token = 
    let (headsQ, headsK, headsV) = computeQKV network indexLayer freqCisRealRow freqCisImagRow token
        keyCacheStep = (keyCache !! stepCount) ++ [headsK]
        valueCacheStep = (valueCache !! stepCount) ++ [headsV]
        keyCache' = replaceAtIndex stepCount keyCacheStep keyCache
        valueCache' = replaceAtIndex stepCount valueCacheStep valueCache
        activations = multiheadActivation network indexLayer keyCache' valueCache' headsQ
        wO = wo (weighting network)
        deltaTokenQKV = matrixVectorMult (wO !! indexLayer) (reshapeMatrixToVector activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = (computeDeltaFFN (weighting network) indexLayer token')
    in (V.zipWith (+) token deltaTokenFFN, keyCache', valueCache')

run :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let
    seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
    network = initModel modelFileContent
    --weighting = checkpointInitWeights network modelFileContent
    (vocab, vocabScores) = tokenizerInit tokenizerFileContent (vocabSize network)
    promptTokens = bpeEncode (T.pack (fromMaybe "" prompt)) vocab vocabScores
    initCache = RunCache { keyCache = [], valueCache = [] }
  textList <- generateTokens network initCache steps promptTokens temperature vocab
  --putStrLn $ "created network: " ++ show (LA.subMatrix (0, 0) (1, 10) (tokenEmbeddingTable (weighting network)))
  --putStrLn $ "created weighting: " ++ show (LA.subMatrix (0, 0) (1, 10) (tokenEmbeddingTable weighting))
  printf "network: # layers %d / # attention heads %d / head dimension %d / vocabulary size %d\n" (nLayers network) (numAttentionHeads network) (headDimension network) (vocabSize network)
  print promptTokens
  print $ map (\token -> vocab !! token) promptTokens
  printf "%d %f\n" seedValue temperature
  putStrLn $ T.unpack $ T.intercalate (T.pack " ") textList
