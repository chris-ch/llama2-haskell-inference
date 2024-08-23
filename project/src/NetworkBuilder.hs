{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}

module NetworkBuilder (
  NetworkConfig(..),
  AttentionKV(..),
  TransformerWeighting(..),
  KeyCache,
  ValueCache,
  Vocabulary,
  PromptTokens,
  Token,
  initModel, tokenizerInit, readVectors
  ) where

import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString as SBS
import qualified Data.Binary.Get as BG
import qualified Data.List as DL
import qualified Data.Vector.Unboxed as V

import Control.Monad (replicateM)
import Data.Maybe (fromMaybe)
import Data.Int (Int32)

import qualified Matrix as M

type KeyCache = [[M.Matrix Float]]
type ValueCache = [[M.Matrix Float]]
type Vocabulary = [BS.ByteString]
type VocabularyScores = [Float]
type Token = Int32
type PromptTokens = [Token]

data AttentionKV = AttentionKV
    { keyCache :: KeyCache
    , valueCache :: ValueCache
    } deriving (Show)

data TransformerWeighting = TransformerWeighting
    { tokenEmbeddingTable :: M.Matrix Float
    , rmsAttWeight :: [V.Vector Float]
    , wq :: [M.Matrix Float]
    , wk :: [M.Matrix Float]
    , wv :: [M.Matrix Float]
    , wo :: [M.Matrix Float]
    , rmsFfnWeight :: [V.Vector Float]
    , w1 :: [M.Matrix Float]
    , w3 :: [M.Matrix Float]
    , w2 :: [M.Matrix Float]
    , rmsFinalWeight :: V.Vector Float
    , freqCisReal :: [V.Vector Float]
    , freqCisImag :: [V.Vector Float]
    } deriving (Show)

data NetworkConfig = NetworkConfig
    { tokenDim :: Int
    , hiddenDim :: Int
    , numLayers :: Int
    , numAttentionHeads :: Int
    , numKeyValueHeads :: Int
    , vocabSize :: Int
    , seqLen :: Int
    , headDimension :: Int
    , weighting :: TransformerWeighting
    } deriving (Show)

readBytes :: Int -> BG.Get BS.ByteString
readBytes count = do
  values <- BG.getByteString count
  return $ SBS.fromStrict values

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
    let bytesPerFloat = 4 :: Int
        totalBytes = count * bytesPerFloat
    bytes <- readBytes totalBytes
    return $ V.unfoldrExactN count (getFloat . BS.splitAt (fromIntegral bytesPerFloat)) bytes
  where
    getFloat :: (BS.ByteString, BS.ByteString) -> (Float, BS.ByteString)
    getFloat (chunk, rest) = (BG.runGet BG.getFloatle chunk, rest)

readVectors :: Int -> Int -> BG.Get [V.Vector Float]
readVectors nrows ncols = replicateM nrows (readVector ncols)

readMatrix :: Int -> Int -> BG.Get (M.Matrix Float)
readMatrix nrows ncols = do
  values <- readVector (nrows * ncols)
  return $ M.fromVector nrows ncols values

readMatrices :: Int -> Int -> Int -> BG.Get [M.Matrix Float]
readMatrices ndepth nrows ncols = replicateM ndepth (readMatrix nrows ncols)

parseNetworkConfigFile :: BG.Get NetworkConfig
parseNetworkConfigFile = do
        tokenDim' <- fromIntegral <$> BG.getInt32le
        hiddenDim' <- fromIntegral <$> BG.getInt32le
        nLayers' <- fromIntegral <$> BG.getInt32le
        numAttentionHeads' <- fromIntegral <$> BG.getInt32le
        numKeyValueHeads' <- fromIntegral <$> BG.getInt32le
        vocabSize' <- fromIntegral <$> BG.getInt32le
        seqLen' <- fromIntegral <$> BG.getInt32le
        tokenEmbeddingTable' <- readMatrix vocabSize' tokenDim'
        rmsAttWeight' <- readVectors nLayers' tokenDim'

        let headSize = tokenDim' `div` numAttentionHeads'

        wq' <- readMatrices nLayers' (numAttentionHeads' * headSize) tokenDim'
        wk' <- readMatrices nLayers' (numKeyValueHeads' * headSize) tokenDim'
        wv' <- readMatrices nLayers' (numKeyValueHeads' * headSize) tokenDim'
        wo' <- readMatrices nLayers' tokenDim' (numAttentionHeads' * headSize)
        rmsFfnWeight' <- readVectors nLayers' tokenDim'
        w1' <- readMatrices nLayers' hiddenDim' tokenDim'
        w2' <- readMatrices nLayers' tokenDim' hiddenDim'
        w3' <- readMatrices nLayers' hiddenDim' tokenDim'
        rmsFinalWeight' <- readVector tokenDim'
        freqCisReal' <- readVectors seqLen' ((tokenDim' `div` (numAttentionHeads')) `div` 2)
        freqCisImag' <- readVectors seqLen' ((tokenDim' `div` (numAttentionHeads')) `div` 2)

        let
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
            { tokenDim = tokenDim'
            , hiddenDim = hiddenDim'
            , numLayers = nLayers'
            , numAttentionHeads = numAttentionHeads'
            , numKeyValueHeads = numKeyValueHeads'
            , vocabSize = abs vocabSize'
            , seqLen = seqLen'
            , headDimension = headSize
            , weighting = weights
            }

initModel :: BS.ByteString -> NetworkConfig
initModel = BG.runGet parseNetworkConfigFile

parseTokens :: BS.ByteString -> Int -> (Vocabulary, VocabularyScores)
parseTokens fileContent size = (vocab, vocabScores)
  where
    scoresTokens = BG.runGet scoresAndTokens fileContent
    vocabScores = fst <$> scoresTokens
    vocab = snd <$> scoresTokens

    scoresAndTokens :: BG.Get [(Float, BS.ByteString)]
    scoresAndTokens = replicateM size readToken

    readToken :: BG.Get (Float, BS.ByteString)
    readToken = do
      score <- BG.getFloatle
      tokenSize <- BG.getInt32le
      token <- BG.getLazyByteString (fromIntegral tokenSize)
      return (score, token)

tokenizerInit :: BS.ByteString -> Int -> BS.ByteString -> (PromptTokens, Vocabulary)
tokenizerInit file size prompt = (bpeEncode prompt vocab vocabScores, vocab)
  where
    (vocab, vocabScores) = parseTokens (BS.drop 4 file) size

strLookup :: BS.ByteString -> Vocabulary -> Int
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
            case strLookup ((vocab !! (fromIntegral tokenPrev)) `BS.append` (vocab !! (fromIntegral tokenNext))) vocab of
              pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, fromIntegral pos)
              _ -> acc

          bestScore :: Float
          bestScore = -1e10

      mergePair :: Int -> Token -> [Token] -> [Token]
      mergePair count token tokens' =
        take count tokens' ++ [token] ++ drop (count + 2) tokens'

bpeEncode :: BS.ByteString -> Vocabulary -> VocabularyScores -> PromptTokens
bpeEncode prompt vocab vocabScores =
  let tokens = map (\char -> fromMaybe (error "Character not found in vocabulary") (DL.elemIndex (BS.pack [char]) vocab)) (BS.unpack prompt)
  in processTokens (map fromIntegral tokens) vocab vocabScores
