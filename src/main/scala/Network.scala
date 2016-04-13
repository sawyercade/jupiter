import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.{sigmoid, exp}

import scala.collection.mutable.ListBuffer
import scala.util.Random
import Network._

/**
 * Created by sawyer on 3/16/16.
 */
class Network(sizes: List[Int]) {
  val numLayers = sizes.length

  val biases : List[DenseVector[Double]] = buildBiases(sizes)

  val weights : List[DenseMatrix[Double]] = buildWeights(sizes)

  def feedForward(inputs : DenseVector[Double]) : DenseVector[Double] = {
    var a : DenseVector[Double] = inputs
    weights.zip(biases).foreach({ case (w, b) => a = sigmoid((w * a) + b) })
    a
  }
}

object Network {
  val RANDOM = new Random(1337)

  /**
   * Builds a vector of node biases for all layers except the input layer
   * @param sizes
   * @return
   */
  def buildBiases(sizes: List[Int]) : List[DenseVector[Double]] = {
    val biases = new ListBuffer[DenseVector[Double]]

    sizes.drop(1).foreach(layerSize =>
      biases += DenseVector.tabulate(layerSize) {i => RANDOM.nextGaussian()} )

    biases.toList
  }

  /**
    * Builds a List[DenseMatrix[Double]] of weights.
    *
    * weights[1] is a sizes(2) by sizes(1) matrix w, where w_j,k is the weight of the connection from node k in layer 1 to node j in layer 2.
    * Although reversing j and k would intuitively make sense, this order allows the vector of activations of layer 2 to be
    * a' = sigmoid(wa + b) where a is the vector of activations of layer 1 and b is the biases vector for layer 1.
    *
    * @param sizes
    * @return
    */
  def buildWeights(sizes: List[Int]) : List[DenseMatrix[Double]] = {
    val weights = new ListBuffer[DenseMatrix[Double]]()

    sizes.dropRight(1).zip(sizes.drop(1)).foreach(
      { case(j,k) =>
        weights += DenseMatrix.tabulate(k, j) { case(a, b) => RANDOM.nextGaussian() }
      })

    weights.toList
  }

  def main(args: Array[String]): Unit = {
    val net = new Network(List(17, 12, 7, 4, 2))

    println("biases:\n" + net.biases.head.toString())
    println("weights(0):\n" + net.weights.head.toString())
  }
}
