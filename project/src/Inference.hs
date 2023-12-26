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
import qualified Data.Vector as V
import qualified Data.Matrix as M

import Control.Monad.State
import Control.Monad (replicateM)
import Data.Binary.Get (runGet, getInt32le, getFloatle)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Vector (Vector)
import Data.Matrix (Matrix)

import Debug.Trace

data RunCache = RunCache
    { keyCache :: [[[Vector Float]]]
    , valueCache :: [[[Vector Float]]]
    } deriving (Show)

update3DList :: [[[a]]] -> Int -> Int -> [a] -> [[[a]]]
update3DList list i j elemList = take i list ++ [update2DList (list !! i) j elemList] ++ drop (i + 1) list

update2DList :: [[a]] -> Int -> [a] -> [[a]]
update2DList list i elemList = take i list ++ [list !! i ++ elemList] ++ drop (i + 1) list

appendStepKey :: State RunCache ()
appendStepKey = modify (\s -> s {keyCache = keyCache s ++ [[]]})

appendStepValue :: State RunCache ()
appendStepValue = modify (\s -> s {valueCache = valueCache s ++ [[]]})

appendStepLayerKey :: Int -> Int -> [Vector Float] -> State RunCache ()
appendStepLayerKey stepCount indexLayer vectors = modify (\s -> s { keyCache = update3DList (keyCache s) stepCount indexLayer vectors })

appendStepLayerValue :: Int -> Int -> [V.Vector Float] -> State RunCache ()
appendStepLayerValue stepCount indexLayer vectors = modify (\s -> s { valueCache = update3DList (valueCache s) stepCount indexLayer vectors })

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
    return $ M.fromLists (DLS.chunksOf nrows values)

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

computeQKV :: Network -> Int -> Vector Float -> Vector Float -> Vector Float -> ([Vector Float], [Vector Float], [Vector Float])
computeQKV network indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token ((rmsAttWeight (weighting network)) !! indexLayer)
    wQ = splitVector (numAttentionHeads network) (matrixVectorMult ((wq (weighting network)) !! indexLayer) rba)
    headsQ = map (\vector -> traceStack ("rotating Q: " ++ show (V.length vector)) applyRotations vector freqCisRealRow freqCisImagRow) wQ
    wK = splitVector (numAttentionHeads network) (matrixVectorMult ((wk (weighting network)) !! indexLayer) rba)
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wK
    headsV = splitVector (numAttentionHeads network) (matrixVectorMult ((wv (weighting network)) !! indexLayer) rba)
  in
    traceStack "computeQKV" (headsQ, headsK, headsV)

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

buildActivation :: Int -> Int -> [[[V.Vector Float]]] -> Int -> [Float] -> Vector Float
buildActivation dimension indexLayer valueCache indexHead headScores =
  V.foldl' (\acc (valueVector, attentionWeight) -> V.zipWith (+) acc (scale attentionWeight valueVector)) zeroVector (V.fromList zippedValues)
  where
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

matrixVectorMult :: (Num a) => Matrix a -> Vector a -> Vector a
matrixVectorMult matrix vector = 
    let result = M.getCol 1 $ M.multStd2 matrix (M.colVector vector)
        traceMessage = "Matrix/Vector multiplication performed. Matrix: (" ++ show (M.nrows matrix) ++ " x " ++ show (M.ncols matrix) ++ "), Vector: " ++ show (V.length vector)
    in traceStack traceMessage result

splitVector :: Int -> V.Vector a -> [V.Vector a]
splitVector m vec = fmap V.fromList $ DLS.chunksOf ((V.length vec) `div` m) (V.toList vec)

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
  in trace "rmsNorm" elementsProduct weights normalized

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
    in traceStack "computeDeltaFFN" matrixVectorMult weight2 (elementsProduct sigmoided hiddenDimensionBuffer2)

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
        activations = traceStack ("headsQ[0] size=" ++ show (V.length (headsQ !! 0))) multiheadActivation network indexLayer keyCache' valueCache' headsQ
        wO = wo (weighting network)
        deltaTokenQKV = traceStack ("activations " ++ show (M.nrows activations) ++ "x" ++ show (M.ncols activations)) matrixVectorMult (wO !! indexLayer) (reshapeMatrixToVector activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = traceStack ("processing FFN layer" ++ show indexLayer) (computeDeltaFFN (weighting network) indexLayer token')
    in traceStack "createLayerToken" (V.zipWith (+) token deltaTokenFFN, keyCache', valueCache')

transformer :: Int -> Int -> Network -> StateT RunCache IO (Vector Float)
transformer tokenCode stepCount network = do
  let token = M.getRow tokenCode (tokenEmbeddingTable (weighting network))
      freqCisRealRow = (freqCisReal (weighting network)) !! stepCount
      freqCisImagRow = (freqCisImag (weighting network)) !! stepCount

  (keyCache, valueCache) <- gets $ \c -> (keyCache c, valueCache c)
  -- Forwarding all the layers
  let (newToken, _, _) =
        DL.foldl' (\(curToken, curKeyCache, curValueCache) indexLayer ->
                     (createLayerToken network stepCount keyCache valueCache indexLayer freqCisRealRow freqCisImagRow curToken))
                  (token, keyCache, valueCache) [0 .. (nLayers network) - 1]

  -- Final rmsnorm
  let finalToken = rmsNorm newToken (rmsFinalWeight (weighting network))

  -- Classifier into logits
  return $ traceStack "createLayerToken" matrixVectorMult (tokenEmbeddingTable (weighting network)) finalToken

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
        cache <- get
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature network vocab 1
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
  (textList, finalState) <- runStateT (generateTokens network steps promptTokens temperature vocab) initCache
  --putStrLn $ "created network: " ++ show (LA.subMatrix (0, 0) (1, 10) (tokenEmbeddingTable (weighting network)))
  --putStrLn $ "created weighting: " ++ show (LA.subMatrix (0, 0) (1, 10) (tokenEmbeddingTable weighting))
  printf "network: # layers %d / # attention heads %d / head dimension %d / vocabulary size %d\n" (nLayers network) (numAttentionHeads network) (headDimension network) (vocabSize network)
  print promptTokens
  print $ map (\token -> vocab !! token) promptTokens
  printf "%d %f\n" seedValue temperature
  putStrLn $ T.unpack $ T.intercalate (T.pack " ") textList
