module CustomRandom where

import Data.Vector
import Data.Matrix
import qualified Data.Vector as V
import qualified Data.Matrix as Mx
import Control.Monad.State
import Control.Monad
import qualified Control.Monad as M
import Data.Bits

-- Define the state type
type StateRNG = Int

-- Define the state monad with StateRNG
type CustomRNG a = State StateRNG a

-- Generating and returning the next value
nextRandomValue :: CustomRNG Float
nextRandomValue = do
  -- Get the current state (custom random value)
  currentValue <- get
  -- Update the state (generate next custom random value)
  let randomValue = customRandom currentValue
  put $ randomValue
  return $ (fromIntegral (randomValue `mod` 1000)) / 1000.0

-- Getting the current value
getRandomValue :: CustomRNG Float
getRandomValue = do
    currentValue <- get
    return $ (fromIntegral (currentValue `mod` 1000)) / 1000.0

seedRandomValue :: Int -> CustomRNG ()
seedRandomValue seed = do
    put $ seed

customRandom :: Int -> Int
customRandom currentValue =
  let valueStep1 = (currentValue * 63) `mod` 0xC4CB7296
      valueStep2 = valueStep1 `xor` 0x1754FBF
      valueStep3 = (valueStep2 * 0xFF) `mod` 4294967296
      valueStep4 = valueStep3 `xor` 0x222F42CB
      valueStep5 = valueStep4 .|. 0x1234567890
  in ((valueStep5 + 14351514) * 32) `mod` 7777333

generateRandomArray :: Int -> CustomRNG [Float]
generateRandomArray size = M.replicateM size nextRandomValue

generateRandomArrays :: Int -> Int -> CustomRNG [[Float]]
generateRandomArrays nrows ncols = M.replicateM nrows (generateRandomArray ncols)

generateRandomVector :: Int -> CustomRNG (Vector Float)
generateRandomVector size = fmap V.fromList $ M.replicateM size nextRandomValue

generateRandomVectors :: Int -> Int -> CustomRNG [Vector Float]
generateRandomVectors count size = M.replicateM count (generateRandomVector size)

generateRandomMatrix :: Int -> Int -> CustomRNG (Matrix Float)
generateRandomMatrix nrows ncols = fmap Mx.fromLists $ generateRandomArrays nrows ncols

generateRandomMatrices :: Int -> Int -> Int -> CustomRNG [Matrix Float]
generateRandomMatrices count nrows ncols = M.replicateM count (generateRandomMatrix nrows ncols)
