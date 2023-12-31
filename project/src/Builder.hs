{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Builder (
  NetworkConfig(..),
  AttentionKV(..), 
  Matrix, 
  TransformerWeighting(..),
  initModel, tokenizerInit, bpeEncode, readVectors
  ) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Text.Encoding as TE
import qualified Data.Binary.Get as BG
import qualified Data.Text as T
import qualified Data.List as DL
import qualified Data.Vector.Unboxed as V

import Control.Monad (replicateM)
import Data.Binary.Get (getInt32le, getFloatle)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Vector.Unboxed (Vector)

type Matrix a = [Vector a] -- Matrix as row vectors

data AttentionKV = AttentionKV
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

data NetworkConfig = NetworkConfig
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

readMatrices :: Int -> Int -> Int -> BG.Get [Matrix Float]
readMatrices ndepth nrows ncols = replicateM ndepth (readVectors nrows ncols)

parseNetworkConfigFile :: BG.Get NetworkConfig
parseNetworkConfigFile = do
        dim <- getInt32le
        hiddenDim <- getInt32le
        nLayers <- getInt32le
        numAttentionHeads <- getInt32le
        numKeyValueHeads <- getInt32le
        vocabSize <- getInt32le
        seqLen <- getInt32le
        tokenEmbeddingTable <- readVectors (fromIntegral vocabSize) (fromIntegral dim)
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
        return $ NetworkConfig
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

initModel :: BSL.ByteString -> NetworkConfig
initModel networkConfigFile = BG.runGet parseNetworkConfigFile networkConfigFile

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
