{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Matrix
  ( Matrix(..)
  , fromVector
  , fromVectors
  , multiplyVector
  , getRowVector
  ) where

import qualified Data.Vector.Unboxed as V

data Matrix a = Matrix
  { numRows :: !Int
  , numCols :: !Int
  , values :: !(V.Vector a)
  } deriving (Eq)

instance (Show a, V.Unbox a) => Show (Matrix a) where
  show :: Matrix a -> String
  show m = "Matrix " ++ show (rows m) ++ "x" ++ show (cols m) ++ ":\n" ++
           unlines [show [get m i j | j <- [0..cols m - 1]] | i <- [0..rows m - 1]]

fromVectors :: V.Unbox a => Int -> Int -> [V.Vector a] -> Matrix a
fromVectors r c vecs = Matrix r c $ V.concat vecs

fromVector :: Int -> Int -> V.Vector a -> Matrix a
fromVector = Matrix

multiplyVector :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector (Matrix r c v) vec = V.generate r $ \i ->
    V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec

getRow :: V.Unbox a => Matrix a -> Int -> [a]
getRow m@(Matrix _ c _) i = [get m i j | j <- [0..c-1]]

getRowVector :: V.Unbox a => Matrix a -> Int -> V.Vector a
getRowVector m = V.fromList . getRow m

get :: V.Unbox a => Matrix a -> Int -> Int -> a
get (Matrix _ c v) i j = v V.! (i * c + j)

rows :: Matrix a -> Int
rows = numRows

cols :: Matrix a -> Int
cols = numCols
