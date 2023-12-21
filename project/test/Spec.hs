module Main where

import Test.Hspec

import qualified InferenceSpec

main :: IO ()
main = hspec $ do
  describe "Inference tests" InferenceSpec.spec
