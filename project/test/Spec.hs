module Main (main) where

import Test.Hspec

import qualified HelperSpec
import qualified InferenceSpec
import qualified TokenGeneratorSpec

main :: IO ()
main = hspec $ do
  describe "Helper tests" HelperSpec.spec
  describe "Inference tests" InferenceSpec.spec
  describe "Token Generator tests" TokenGeneratorSpec.spec
