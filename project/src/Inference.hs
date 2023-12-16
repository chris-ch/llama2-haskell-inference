{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE LambdaCase #-}

module Inference( run ) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Binary.Get as BG
import qualified Data.Text.Encoding as TE
import qualified Data.Text as T

import System.Random
import Numeric.LinearAlgebra
import Data.Array

import Control.Monad (replicateM)
import Control.Monad.IO.Class (liftIO)
import Data.Binary.Get (runGet, getInt32le, getWord32le, getFloatle)
import Text.Printf (printf)
import Data.Maybe (fromMaybe)
import Data.Text (Text, pack, unpack, isAscii)
import Data.Text.Encoding.Error (lenientDecode)
import Data.Word (Word32)

data TransformerWeighting = TransformerWeighting
    { tokenEmbeddingTable :: Array Int Float
    , rmsAttWeight :: Array Int Float
    , wq :: Array Int Float
    , wk :: Array Int Float
    , wv :: Array Int Float
    , wo :: Array Int Float
    , rmsFfnWeight :: Array Int Float
    , w1 :: Array Int Float
    , w3 :: Array Int Float
    , w2 :: Array Int Float
    , rmsFinalWeight :: Array Int Float
    , freqCisReal :: Array Int (Array Int Float)
    , freqCisImag :: Array Int (Array Int Float)
    } deriving (Show)

data Network = Network
    { dim :: Int
    , hiddenDim :: Int
    , nLayers :: Int
    , numAttentionHeads :: Int
    , numKeyValueHeads :: Int
    , vocabSize :: Int
    , seqLen :: Int
    , weighting :: Maybe TransformerWeighting
    , headDimension :: Int
    } deriving (Show)

data RunState = RunState
    { scores :: Array Int (Array Int Float) -- scores/attention values (n_heads, seq_len)
    , keyCache :: Array Int (Array Int (Array Int Float)) -- (layer, seq_len, dim)
    , valueCache :: Array Int (Array Int (Array Int Float)) -- (layer, seq_len, dim)
    } deriving (Show)


rmsNorm :: Vector Double -> Vector Double -> Vector Double
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
        , weighting = Nothing  -- Replace with actual initialization logic
        , headDimension = fromIntegral dim `div` fromIntegral numAttentionHeads
        }
        where
          (dim, hiddenDim, nLayers, numAttentionHeads, numKeyValueHeads, vocabSize, seqLen) =
            runGet (
              (\a b c d e f g -> (a, b, c, d, e, f, g)) 
              <$> getInt32le <*> getInt32le <*> getInt32le <*> getInt32le <*> getInt32le
              <*> getInt32le <*> getInt32le) networkConfigFile

readToken :: BSL.ByteString -> (Float, String)
readToken file = (score, str)
  where
    score = BG.runGet BG.getFloatle (BSL.take 4 file)
    length = BG.runGet BG.getInt32le (BSL.take 4 file)
    bstr = TE.decodeUtf8 $ BSL.toStrict $ BSL.take (fromIntegral length) file
    str = unpack bstr

parseTokens :: BSL.ByteString -> Int -> Int -> ([String], [Float], Int)
parseTokens file maxTokenLength size = (vocab, vocabScores, maxTokenLength)
  where
    readTemp2 :: BG.Get (Float, String)
    readTemp2 = readToken <$> pure file
    scoresAndStrings = BG.runGet (replicateM size readTemp2) file

    vocabScores = fst <$> scoresAndStrings
    vocab = snd <$> scoresAndStrings
  
tokenizerInit :: BSL.ByteString -> Int -> ([String], [Float], Int)
tokenizerInit file size = parseTokens file maxTokenLength size
  where 
        extractMaxTokenLength :: BSL.ByteString -> Int
        extractMaxTokenLength file = fromIntegral $ runGet getWord32le (BSL.take 4 file)

        maxTokenLength = extractMaxTokenLength file


run :: BSL.ByteString -> BSL.ByteString -> Double -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
  network <- loadNetwork modelFileContent
  let (vocab, vocabScores, maxTokenLength) = tokenizerInit tokenizerFileContent (vocabSize network)

  putStrLn $ "created network: " ++ show network
  print vocab
  printf "%d %f\n" seedValue temperature
