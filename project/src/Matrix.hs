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

fromList :: V.Unbox a => Int -> Int -> [a] -> Matrix a
fromList r c xs
  | length xs == r * c = Matrix r c (V.fromList xs)
  | otherwise = error $ "Input list length (" ++ show (length xs) ++ ") doesn't match matrix dimensions (" ++ show r ++ "x" ++ show c ++ ")"

fromVectors :: V.Unbox a => Int -> Int -> [V.Vector a] -> Matrix a
fromVectors r c vecs = fromVector r c $ V.concat vecs

fromVector :: V.Unbox a => Int -> Int -> V.Vector a -> Matrix a
fromVector r c xs
  | V.length xs == r * c = Matrix r c xs
  | otherwise = error $ "Input vector length (" ++ show (V.length xs) ++ ") doesn't match matrix dimensions (" ++ show r ++ "x" ++ show c ++ ")"

toList :: V.Unbox a => Matrix a -> [a]
toList = V.toList . values

multiplyVector :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

getRow :: V.Unbox a => Matrix a -> Int -> [a]
getRow m@(Matrix r c _) i
  | i < 0 || i >= r = error "Row index out of bounds"
  | otherwise = [get m i j | j <- [0..c-1]]

getRowVector :: V.Unbox a => Matrix a -> Int -> V.Vector a
getRowVector m = V.fromList . getRow m

getCol :: V.Unbox a => Matrix a -> Int -> [a]
getCol m@(Matrix r c _) j
  | j < 0 || j >= c = error "Column index out of bounds"
  | otherwise = [get m i j | i <- [0..r-1]]

getColVector :: V.Unbox a => Matrix a -> Int -> V.Vector a
getColVector m = V.fromList . getCol m

get :: V.Unbox a => Matrix a -> Int -> Int -> a
get (Matrix r c v) i j
  | i < 0 || i >= r || j < 0 || j >= c = error "Index out of bounds"
  | otherwise = v V.! (i * c + j)

set :: V.Unbox a => Matrix a -> Int -> Int -> a -> Matrix a
set (Matrix r c v) i j x
  | i < 0 || i >= r || j < 0 || j >= c = error "Index out of bounds"
  | otherwise = Matrix r c (v V.// [(i * c + j, x)])

rows :: Matrix a -> Int
rows = numRows

cols :: Matrix a -> Int
cols = numCols

shape :: Matrix a -> (Int, Int)
shape m = (rows m, cols m)
