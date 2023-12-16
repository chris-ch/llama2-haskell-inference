{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
module Inference( run ) where

import Numeric.LinearAlgebra

import System.Random
import Text.Printf (printf)
import Control.Monad
import qualified Data.ByteString.Lazy as BSL
import Data.Maybe (fromMaybe)


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
