{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Inference( run ) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Binary.Get as BG
import qualified Data.Text.Encoding as TE
import qualified Data.Text as T
import qualified Data.Array as A
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified Numeric.LinearAlgebra as LA


import System.Random
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
    } deriving (Show)

data RunState = RunState
    { scores :: LA.Matrix Float -- scores/attention values (n_heads, seq_len)
    , keyCache :: Array (Int, Int, Int, Int) Float
    , valueCache :: Array (Int, Int, Int, Int) Float
    } deriving (Show)


rmsNorm :: LA.Vector Double -> LA.Vector Double -> LA.Vector Double
rmsNorm x weight =
  let ss = (sumElements (x^2) / fromIntegral (size x)) + 1e-5
      normalized = cmap (* (1.0 / sqrt ss)) x
  in weight * normalized

entryPoint :: Int -> IO ()
entryPoint count = do
  let x = vector [1.0, 2.0, 3.0]
      w = vector [0.1, 0.2, 0.3]
      result = rmsNorm x w
  print result

loadNetwork :: BSL.ByteString -> IO Network
loadNetwork networkConfigFile = return Network
        { dim = fromIntegral dim
        , hiddenDim = fromIntegral hiddenDim
        , nLayers = fromIntegral nLayers
        , numAttentionHeads = fromIntegral numAttentionHeads
        , numKeyValueHeads = fromIntegral numKeyValueHeads
        , vocabSize = fromIntegral $ abs vocabSize
        , seqLen = fromIntegral seqLen
        , headDimension = fromIntegral dim `div` fromIntegral numAttentionHeads
        }
        where
          (dim, hiddenDim, nLayers, numAttentionHeads, numKeyValueHeads, vocabSize, seqLen) =
            runGet (
              (\a b c d e f g -> (a, b, c, d, e, f, g)) 
              <$> getInt32le <*> getInt32le <*> getInt32le <*> getInt32le <*> getInt32le
              <*> getInt32le <*> getInt32le) networkConfigFile

readAsArray :: Int -> BG.Get (LA.Vector Float)
readAsArray count = do
    bytes <- replicateM (count * 4) getFloatle
    return $ LA.fromList bytes

readMatrix :: Int -> Int -> BG.Get (LA.Matrix Float)
readMatrix nrows ncols = do
    bytes <- replicateM (nrows * ncols * 4) getFloatle
    return $ LA.fromLists (DLS.chunksOf ncols bytes)

readMatrices :: Int -> Int -> Int -> BG.Get [LA.Matrix Float]
readMatrices ndepth nrows ncols = do
    bytes <- replicateM (nrows * ncols * ndepth * 4) getFloatle
    let chunks = DLS.chunksOf (nrows * ncols) bytes
        matrices = map (\chunk -> LA.reshape ncols (LA.fromList chunk)) chunks
    return matrices

checkpointInitWeights :: Network -> BSL.ByteString -> TransformerWeighting
checkpointInitWeights conf file =
    let nrows = vocabSize conf
        ncols = dim conf
        nLayers' = nLayers conf
        hiddenDim' = hiddenDim conf
        headDimension' = headDimension conf
        seqLen' = seqLen conf
    in runGet (do
        tokenEmbeddingTable <- readMatrix nrows ncols
        rmsAttWeight <- readMatrix nLayers' ncols
        wq <- readMatrices nLayers' ncols ncols
        wk <- readMatrices nLayers' ncols ncols
        wv <- readMatrices nLayers' ncols ncols
        wo <- readMatrices nLayers' ncols ncols
        rmsFfnWeight <- readMatrix nLayers' ncols
        w1 <- readMatrices nLayers' ncols hiddenDim'
        w2 <- readMatrices nLayers' hiddenDim' ncols
        w3 <- readMatrices nLayers' ncols hiddenDim'
        rmsFinalWeight <- readAsArray ncols
        freqCisReal <- readMatrix seqLen' (headDimension' `div` 2)
        freqCisImag <- readMatrix seqLen' (headDimension' `div` 2)
        return $ TransformerWeighting
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
            }) file

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

run :: BSL.ByteString -> BSL.ByteString -> Double -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
  network <- loadNetwork modelFileContent
  let
    weighting = checkpointInitWeights network modelFileContent
    (vocab, vocabScores) = tokenizerInit tokenizerFileContent (vocabSize network)
    state = makeInitState network
    promptTokens = bpeEncode (T.pack (fromMaybe "" prompt)) vocab vocabScores

  putStrLn $ "created network: " ++ show network
  print promptTokens
  print $ map (\token -> vocab !! token) promptTokens
  printf "%d %f\n" seedValue temperature
