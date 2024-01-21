{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}

module NetworkBuilder (
  NetworkConfig(..),
  AttentionKV(..),
  Matrix,
  TransformerWeighting(..),
  KeyCache,
  ValueCache,
  Vocabulary,
  PromptTokens,
  Token,
  initModel, tokenizerInit, readVectors
  ) where

import qualified Data.ByteString.Lazy as BS
import qualified Data.Text.Encoding as TE
import qualified Data.Binary.Get as BG
import qualified Data.Text as T
import qualified Data.List as DL
import qualified Data.Vector.Unboxed as V

import Control.Monad (replicateM)
import Data.Binary.Get (getInt32le, getFloatle)
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Int (Int32)
import Data.Vector.Unboxed (Vector)

type Matrix a = [Vector a] -- Matrix as row vectors
type KeyCache = [[Matrix Float]]
type ValueCache = [[Matrix Float]]
type Vocabulary = [T.Text]
type VocabularyScores = [Float]
type Token = Int32
type PromptTokens = [Token]

data AttentionKV = AttentionKV
    { keyCache :: KeyCache
    , valueCache :: ValueCache
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
        dim' <- fromIntegral <$> getInt32le
        hiddenDim' <- fromIntegral <$> getInt32le
        nLayers' <- fromIntegral <$> getInt32le
        numAttentionHeads' <- fromIntegral <$> getInt32le
        numKeyValueHeads' <- fromIntegral <$> getInt32le
        vocabSize' <- fromIntegral <$> getInt32le
        seqLen' <- fromIntegral <$> getInt32le
        tokenEmbeddingTable' <- readVectors vocabSize' dim'
        rmsAttWeight' <- readVectors nLayers' dim'
        wq' <- readMatrices nLayers' dim' dim'
        wk' <- readMatrices nLayers' dim' dim'
        wv' <- readMatrices nLayers' dim' dim'
        wo' <- readMatrices nLayers' dim' dim'
        rmsFfnWeight' <- readVectors nLayers' dim'
        w1' <- readMatrices nLayers' hiddenDim' dim'
        w2' <- readMatrices nLayers' dim' hiddenDim'
        w3' <- readMatrices nLayers' hiddenDim' dim'
        rmsFinalWeight' <- readVector dim'
        freqCisReal' <- readVectors seqLen' ((dim' `div` (numAttentionHeads')) `div` 2)
        freqCisImag' <- readVectors seqLen' ((dim' `div` (numAttentionHeads')) `div` 2)

        let
            headDim = dim' `div` numAttentionHeads'
            weights = TransformerWeighting
              { tokenEmbeddingTable = tokenEmbeddingTable'
              , rmsAttWeight = rmsAttWeight'
              , wq = wq'
              , wk = wk'
              , wv = wv'
              , wo = wo'
              , rmsFfnWeight = rmsFfnWeight'
              , w1 = w1'
              , w2 = w2'
              , w3 = w3'
              , rmsFinalWeight = rmsFinalWeight'
              , freqCisReal = freqCisReal'
              , freqCisImag = freqCisImag'
              }
        return $ NetworkConfig
            { dim = dim'
            , hiddenDim = hiddenDim'
            , nLayers = nLayers'
            , numAttentionHeads = numAttentionHeads'
            , numKeyValueHeads = numKeyValueHeads'
            , vocabSize = abs vocabSize'
            , seqLen = seqLen'
            , headDimension = headDim
            , weighting = weights
            }

initModel :: BS.ByteString -> NetworkConfig
initModel = BG.runGet parseNetworkConfigFile

parseTokens :: BS.ByteString -> Int -> (Vocabulary, VocabularyScores)
parseTokens file size = (vocab, vocabScores)
  where
    readToken :: BG.Get (Float, T.Text)
    readToken = do
      score <- BG.getFloatle
      tokenSize <- BG.getInt32le
      bstr <- TE.decodeUtf8 . BS.toStrict <$> BG.getLazyByteString (fromIntegral tokenSize)
      return (score, bstr)

    scoresAndStrings :: BG.Get [(Float, T.Text)]
    scoresAndStrings = replicateM size readToken

    vocabScores = fst <$> BG.runGet scoresAndStrings file
    vocab = snd <$> BG.runGet scoresAndStrings file

tokenizerInit :: BS.ByteString -> Int -> String -> (PromptTokens, Vocabulary)
tokenizerInit file size prompt= (bpeEncode (T.pack prompt) vocab vocabScores, vocab)
  where
    (vocab, vocabScores) = parseTokens (BS.drop 4 file) size

strLookup :: Text -> Vocabulary -> Int
strLookup occurrence = fromMaybe (-1) . DL.elemIndex occurrence

processTokens :: [Token] -> Vocabulary -> VocabularyScores -> PromptTokens
processTokens tokens vocab vocabScores = case findBestPair tokens of
        Just (bestIndex, bestToken) ->
          processTokens (mergePair bestIndex bestToken tokens) vocab vocabScores
        Nothing ->
          tokens
    where
      findBestPair :: [Token] -> Maybe (Int, Token)
      findBestPair tokens' = foldr checkPair Nothing (zip [0..] (zip tokens' (drop 1 tokens')))
        where
          checkPair :: (Int, (Token, Token)) -> Maybe (Int, Token) -> Maybe (Int, Token)
          checkPair (count, (tokenPrev, tokenNext)) acc =
            case strLookup ((vocab !! (fromIntegral tokenPrev)) `T.append` (vocab !! (fromIntegral tokenNext))) vocab of
              pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, fromIntegral pos)
              _ -> acc

          bestScore :: Float
          bestScore = -1e10

      mergePair :: Int -> Token -> [Token] -> [Token]
      mergePair count token tokens' =
        take count tokens' ++ [token] ++ drop (count + 2) tokens'

bpeEncode :: T.Text -> Vocabulary -> VocabularyScores -> PromptTokens
bpeEncode prompt vocab vocabScores =
  let tokens = map (\char -> fromMaybe (error "Character not found in vocabulary") (DL.elemIndex (T.pack [char]) vocab)) (T.unpack prompt)
  in processTokens (map fromIntegral tokens) vocab vocabScores
