module Lib
    ( entryPoint
    ) where

import System.Random
import Text.Printf
import Control.Monad

choose :: (a, a) -> IO a
choose (x, y) = do
  r <- randomIO :: IO Bool
  if r then return x else return y

data Gender = Boy | Girl deriving (Eq, Ord, Enum, Show)

children :: IO (Gender, Gender)
children = do
    child1 <- choose (Boy, Girl)
    child2 <- choose (Boy, Girl)
    return (child1, child2)

rule :: (Int, Int) -> IO (Gender, Gender) -> IO (Int, Int)
rule (countHits, countAll) genders = do
  (child1, child2) <- genders
  return $ case (child1, child2) of
      (Boy, Boy) ->  (countHits, countAll + 1)
      (Boy, Girl) ->  (countHits + 1, countAll + 1)
      (Girl, Boy) ->  (countHits + 1, countAll + 1)
      (Girl, Girl) ->  (countHits, countAll)

stats :: Int -> IO (Int, Int)
stats count = foldM rule (0, 0) [children | _ <- [1..count]]

entryPoint :: IO ()
entryPoint = do
  value <- stats 10000
  printf "%d %d --> %.2f\n" (fst value) (snd value) (fromIntegral (fst value) / fromIntegral (snd value) :: Float)
