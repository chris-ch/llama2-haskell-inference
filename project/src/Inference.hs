{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
module Inference( run ) where


import qualified Data.ByteString.Lazy as BSL
import System.Random
import Control.Monad
import Numeric.LinearAlgebra
import Data.Array
import Text.Printf (printf)
import Data.Maybe (fromMaybe)

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

run :: BSL.ByteString -> BSL.ByteString -> Double -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  let seedValue = fromMaybe 0 seed -- Provide a default value if seed is Nothing
  printf "%d %f" seedValue temperature
