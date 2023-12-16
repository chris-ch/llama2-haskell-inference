{-# LANGUAGE FlexibleContexts #-}
module Lib
    ( entryPoint
    ) where

import Numeric.LinearAlgebra

import System.Random
import Text.Printf
import Control.Monad
import qualified Data.List as L (intercalate)

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
