{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Matrix
  ( Matrix(..)
  , fromList
  , fromVector
  , fromVectors
  , fromVectors1
  , fromVectors2
  , toList
  , get
  , set
  , rows
  , cols
  , shape
  , multiplyVector
  , multiplyVector1
  , multiplyVector2
  , multiplyVector3
  , multiplyVector4
  , multiplyVector5
  , multiplyVector6
  , multiplyVector7
  , getRow
  , getCol
  , getCol1
  , getCol2
  , getCol3
  ) where

import qualified Data.Vector.Unboxed as V

data Matrix a = Matrix
  { matRows :: !Int
  , matCols :: !Int
  , matData :: !(V.Vector a)
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

fromVectors1 :: V.Unbox a => Int -> Int -> [V.Vector a] -> Matrix a
fromVectors1 r c vecs = fromVector1 r c $ V.concat vecs

fromVectors2 :: V.Unbox a => Int -> Int -> [V.Vector a] -> Matrix a
fromVectors2 r c vecs = fromVector2 r c $ V.concat vecs

fromVector :: V.Unbox a => Int -> Int -> V.Vector a -> Matrix a
fromVector r c xs
  | V.length xs == r * c = Matrix r c xs
  | otherwise = error $ "Input vector length (" ++ show (V.length xs) ++ ") doesn't match matrix dimensions (" ++ show r ++ "x" ++ show c ++ ")"

fromVector1 :: V.Unbox a => Int -> Int -> V.Vector a -> Matrix a
fromVector1 r c xs
  | V.length xs == r * c = Matrix r c xs
  | otherwise = error "Input list length doesn't match matrix dimensions"

fromVector2 :: V.Unbox a => Int -> Int -> V.Vector a -> Matrix a
fromVector2 r c xs
  | V.length xs == r * c = Matrix r c xs
  | otherwise = error "Input list length doesn't match matrix dimensions"

toList :: V.Unbox a => Matrix a -> [a]
toList = V.toList . matData

multiplyVector :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector1 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector1 m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector2 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector2 m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector3 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector3 m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector4 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector4 m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector5 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector5 m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector6 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector6 m@(Matrix r c v) vec
  | V.length vec /= c = error $ "Vector length (" ++ show (V.length vec) ++ ") must match matrix (" ++ show r ++ "," ++ show c ++ ") column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

multiplyVector7 :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector7 m@(Matrix r c v) vec
  | V.length vec /= c = error "Vector length must match matrix column count"
  | otherwise = V.generate r (\i -> V.sum $ V.zipWith (*) (V.slice (i * c) c v) vec)

getRow :: V.Unbox a => Matrix a -> Int -> [a]
getRow m@(Matrix r c _) i
  | i < 0 || i >= r = error "Row index out of bounds"
  | otherwise = [get m i j | j <- [0..c-1]]

getCol :: V.Unbox a => Matrix a -> Int -> [a]
getCol m@(Matrix r c _) j
  | j < 0 || j >= c = error "Column index out of bounds"
  | otherwise = [get m i j | i <- [0..r-1]]

getCol1 :: V.Unbox a => Matrix a -> Int -> [a]
getCol1 m@(Matrix r c _) j
  | j < 0 || j >= c = error "Column index out of bounds"
  | otherwise = [get m i j | i <- [0..r-1]]

getCol2 :: V.Unbox a => Matrix a -> Int -> [a]
getCol2 m@(Matrix r c _) j
  | j < 0 || j >= c = error "Column index out of bounds"
  | otherwise = [get m i j | i <- [0..r-1]]

getCol3 :: V.Unbox a => Matrix a -> Int -> [a]
getCol3 m@(Matrix r c _) j
  | j < 0 || j >= c = error "Column index out of bounds"
  | otherwise = [get m i j | i <- [0..r-1]]

get :: V.Unbox a => Matrix a -> Int -> Int -> a
get (Matrix r c v) i j
  | i < 0 || i >= r || j < 0 || j >= c = error "Index out of bounds"
  | otherwise = v V.! (i * c + j)

set :: V.Unbox a => Matrix a -> Int -> Int -> a -> Matrix a
set (Matrix r c v) i j x
  | i < 0 || i >= r || j < 0 || j >= c = error "Index out of bounds"
  | otherwise = Matrix r c (v V.// [(i * c + j, x)])

rows :: Matrix a -> Int
rows = matRows

cols :: Matrix a -> Int
cols = matCols

shape :: Matrix a -> (Int, Int)
shape m = (rows m, cols m)
