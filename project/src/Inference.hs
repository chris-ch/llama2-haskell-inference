{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Inference( run ) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Binary.Get as BG
import qualified Data.Text.Encoding as TE
import qualified Data.Text as T
import qualified Data.Char as C
import qualified Data.Array as A
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified Numeric.LinearAlgebra as LA
import qualified System.Random as R

import Data.Array (Array, array, range)
import Numeric.LinearAlgebra (konst, sumElements, cmap, size, vector)
import Control.Monad (replicateM)
import Control.Monad.IO.Class (liftIO)
import Data.Binary.Get (runGet, getInt32le, getWord32le, getFloatle)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Text (Text, isAscii)
import Data.Text.Encoding.Error (lenientDecode)
import Data.Word (Word32)

data RunState = RunState
    { scores :: LA.Matrix Float -- scores/attention values (n_heads, seq_len)
    , keyCache :: Array (Int, Int, Int, Int) Float
    , valueCache :: Array (Int, Int, Int, Int) Float
    } deriving (Show)

data TransformerWeighting = TransformerWeighting
    { tokenEmbeddingTable :: LA.Matrix Float
    , rmsAttWeight :: LA.Matrix Float
    , wq :: [LA.Matrix Float]
    , wk :: [LA.Matrix Float]
    , wv :: [LA.Matrix Float]
    , wo :: [LA.Matrix Float]
    , rmsFfnWeight :: LA.Matrix Float
    , w1 :: [LA.Matrix Float]
    , w3 :: [LA.Matrix Float]
    , w2 :: [LA.Matrix Float]
    , rmsFinalWeight :: LA.Vector Float
    , freqCisReal :: LA.Matrix Float
    , freqCisImag :: LA.Matrix Float
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

rmsNorm :: LA.Vector Float -> LA.Vector Float -> LA.Vector Float
rmsNorm vector weight =
  let ss = (sumElements (vector^2) / fromIntegral (size vector)) + 1e-5
      normalized = cmap (* (1.0 / sqrt ss)) vector
  in weight * normalized

softmax :: LA.Vector Float -> Int -> LA.Vector Float
softmax values size = softmaxValues <> (LA.subVector size (LA.size values - size) values)
  where
    maxVal = LA.maxElement $ LA.subVector 0 size values
    expValues = LA.cmap exp (LA.subVector 0 size values - LA.scalar maxVal)
    sumExpValues = LA.sumElements expValues
    softmaxValues = LA.cmap (/ sumExpValues) expValues

drawSample :: LA.Vector Float -> IO Int
drawSample probabilities = do
  r <- R.randomIO :: IO Float
  let cdf = DL.scanl1 (+) (LA.toList probabilities)
  return $ go cdf r 0
  where
    go (p:ps) r acc
      | r < p = acc
      | otherwise = go ps r (acc + 1)
    go _ _ acc = acc

readVector :: Int -> BG.Get (LA.Vector Float)
readVector count = do
    values <- replicateM (count) getFloatle
    return $ LA.fromList values

readMatrix :: Int -> Int -> BG.Get (LA.Matrix Float)
readMatrix nrows ncols = do
    values <- replicateM (nrows * ncols) getFloatle
    return $ LA.fromLists (DLS.chunksOf ncols values)

readMatrices :: Int -> Int -> Int -> BG.Get [LA.Matrix Float]
readMatrices ndepth nrows ncols = do
    values <- replicateM (nrows * ncols * ndepth) getFloatle
    let chunks = DLS.chunksOf (nrows * ncols) values
        matrices = map (\chunk -> LA.reshape ncols (LA.fromList chunk)) chunks
    return matrices

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
        rmsAttWeight <- readMatrix (fromIntegral nLayers) (fromIntegral dim)
        wq <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        wk <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        wv <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        wo <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        rmsFfnWeight <- readMatrix (fromIntegral nLayers) (fromIntegral dim)
        w1 <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral dim)
        w2 <- readMatrices (fromIntegral nLayers) (fromIntegral hiddenDim) (fromIntegral dim)
        w3 <- readMatrices (fromIntegral nLayers) (fromIntegral dim) (fromIntegral hiddenDim)
        rmsFinalWeight <- readVector (fromIntegral dim)
        freqCisReal <- readMatrix (fromIntegral seqLen) (((fromIntegral dim) `div` (fromIntegral numAttentionHeads)) `div` 2)
        freqCisImag <- readMatrix (fromIntegral seqLen) (((fromIntegral dim) `div` (fromIntegral numAttentionHeads)) `div` 2)

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
      length <- BG.getInt32le
      bstr <- TE.decodeUtf8 . BSL.toStrict <$> BG.getLazyByteString (fromIntegral length)
      return (score, bstr)

    scoresAndStrings :: BG.Get [(Float, T.Text)]
    scoresAndStrings = replicateM size readToken

    vocabScores = fst <$> BG.runGet scoresAndStrings file
    vocab = snd <$> BG.runGet scoresAndStrings file

tokenizerInit :: BSL.ByteString -> Int -> ([T.Text], [Float])
tokenizerInit file size = parseTokens (BSL.drop 4 file) size

makeInitState :: Network -> RunState
makeInitState network = RunState
  { scores = konst (0::Float) (numAttentionHeads network, seqLen network) :: LA.Matrix Float
  , keyCache = array bounds [(index, 0::Float) | index <- range bounds]
  , valueCache = array bounds [(index, 0::Float) | index <- range bounds]
  } where
      bounds = ((0, 0, 0, 0), (seqLen network - 1, nLayers network - 1, numAttentionHeads network - 1, headDimension network - 1))

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
            pos | pos /= -1 && vocabScores !! pos > bestScore ->
              Just (count, pos)
            _ ->
              acc

        bestScore :: Float
        bestScore = -1e10

    mergePair :: Int -> Int -> [Int] -> [Int]
    mergePair idx id tokens' =
      take idx tokens' ++ [id] ++ drop (idx + 2) tokens'

bpeEncode :: T.Text -> [T.Text] -> [Float] -> [Int]
bpeEncode prompt vocab vocabScores =
  let tokens = map (\char -> fromMaybe (error "Character not found in vocabulary") (DL.elemIndex (T.pack [char]) vocab)) (T.unpack prompt)
  in processTokens tokens vocab vocabScores

transformer :: Int -> Int -> Network -> RunState -> LA.Vector Float
transformer tokenCode timestep network state =
  -- Replace this with your actual implementation
  LA.fromList (replicate (vocabSize network) 1.0)

generate :: Network -> RunState -> Int -> [Int] -> Float -> [Text] -> IO [Text]
generate network state checkedMaxSteps promptTokens temperature vocab =
  go 0 []
  where
    go timestep result
      | timestep >= checkedMaxSteps = return result
      | otherwise = do
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature network vocab 1 state
        go (timestep + 1) (result ++ [tokenStr | nextToken /= 1])

generateNextToken :: Int -> [Int] -> Float -> Network -> [Text] -> Int -> RunState -> IO (Text, Int)
generateNextToken timestep promptTokens temperature network vocab tokenCode state = do
  let logits = transformer tokenCode timestep network state
  nextToken <- if timestep < length promptTokens
    then return (promptTokens !! timestep)
    else if temperature == 0.0
      then return (LA.maxIndex logits)
      else drawSample $ softmax (LA.cmap (/ temperature) logits) (vocabSize network)
  let tokenStr =
        if tokenCode == 1 && C.isSpace (T.head (vocab !! nextToken))
          then T.tail (vocab !! nextToken)
          else vocab !! nextToken
  return (tokenStr, nextToken)

run :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let
    seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
    network = initModel modelFileContent
    --weighting = checkpointInitWeights network modelFileContent
    (vocab, vocabScores) = tokenizerInit tokenizerFileContent (vocabSize network)
    state = makeInitState network
    promptTokens = bpeEncode (T.pack (fromMaybe "" prompt)) vocab vocabScores
  result <- generate network state steps promptTokens temperature vocab
  putStrLn $ "created network: " ++ show (LA.subMatrix (0, 0) (1, 10) (tokenEmbeddingTable (weighting network)))
  --putStrLn $ "created weighting: " ++ show (LA.subMatrix (0, 0) (1, 10) (tokenEmbeddingTable weighting))
  print promptTokens
  print $ map (\token -> vocab !! token) promptTokens
  printf "%d %f\n" seedValue temperature
  print result
