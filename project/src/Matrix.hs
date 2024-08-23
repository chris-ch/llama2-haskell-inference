{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE BangPatterns #-}

module Matrix
  ( Matrix(..)
  , fromVectors
  , multiplyVector
  , multiplyVectorInPlace
  , getRowVector
  , getRowVectors
  ) where

import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import Control.Monad.Primitive (PrimMonad, PrimState)
import Control.Monad (forM_)

data Matrix a = Matrix
  { numRows :: !Int
  , numCols :: !Int
  , values :: !(V.Vector a)
  } deriving (Eq)

instance (Show a, V.Unbox a) => Show (Matrix a) where
  show :: Matrix a -> String
  show m = "Matrix " ++ show (numRows m) ++ "x" ++ show (numCols m) ++ ":\n" ++
           unlines [show [get m i j | j <- [0..numCols m - 1]] | i <- [0..numRows m - 1]]

fromVectors :: V.Unbox a => Int -> Int -> [V.Vector a] -> Matrix a
fromVectors r c vecs = Matrix r c $ V.concat vecs

multiplyVector :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector m@(Matrix r _ _) vec = V.generate r $ \i ->
    V.sum $ V.zipWith (*) (getRowVector m i) vec

multiplyVectorInPlace :: (PrimMonad m, V.Unbox a, Num a) 
                      => Matrix a 
                      -> V.Vector a 
                      -> MV.MVector (PrimState m) a 
                      -> m ()
multiplyVectorInPlace (Matrix rows cols matrixData) vec result = do
    forM_ [0..rows-1] $ \i -> do
        let row = V.slice (i * cols) cols matrixData
        !total <- V.foldM' (\acc (a, b) -> return $! acc + a * b) 0 $ V.zip row vec
        MV.write result i total


-- Usage example:
multiplyVector' :: (V.Unbox a, Num a) => Matrix a -> V.Vector a -> V.Vector a
multiplyVector' m@(Matrix rows _ _) vec = 
    V.create $ do
        result <- MV.new rows
        multiplyVectorInPlace m vec result
        return result

getRowVector :: V.Unbox a => Matrix a -> Int -> V.Vector a
getRowVector (Matrix _ c v) rowIndex = V.slice (rowIndex * c) c v

splitEvery :: V.Unbox a => Int -> V.Vector a -> [V.Vector a]
splitEvery n vec
    | V.null vec = []
    | otherwise = chunk : splitEvery n rest
  where
    (chunk, rest) = V.splitAt n vec

getRowVectors :: V.Unbox a => Matrix a -> [V.Vector a]
getRowVectors (Matrix _ cols vals) = splitEvery cols vals

get :: V.Unbox a => Matrix a -> Int -> Int -> a
get (Matrix _ c v) i j = v V.! (i * c + j)
