package org.example;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Clustering {
    public static void main(String[] args) {
        String APP_NAME = "Cluster";
        String APP_MASTER = "local";
        SparkConf sconf = new SparkConf().setAppName(APP_NAME).setMaster(APP_MASTER);
        SparkSession spark = SparkSession.builder()
                .config(sconf)
                .getOrCreate();
// Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm").load("sample_kmeans_data.txt");

// Trains a k-means model.
        KMeans kmeans = new KMeans().setK(2).setSeed(1L);
        KMeansModel model = kmeans.fit(dataset);

// Make predictions
        Dataset<Row> predictions = model.transform(dataset);

// Evaluate clustering by computing Silhouette score
        ClusteringEvaluator evaluator = new ClusteringEvaluator();

        double silhouette = evaluator.evaluate(predictions);
        System.out.println("Silhouette with squared euclidean distance = " + silhouette);

// Shows the result.
        Vector[] centers = model.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }

    }
}