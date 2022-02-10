package github;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class crossValidationNaive {
	
	static long startTime = System.nanoTime();
	
	public static void main(String[] args) throws Exception {
		DataSource sorDataSource = new DataSource(args[0]);
		Instances dataInstances = sorDataSource.getDataSet();
		
		dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
		
		NaiveBayes modeBayes = new NaiveBayes();
		Evaluation evaluation = new Evaluation(dataInstances);
		evaluation.crossValidateModel(modeBayes, dataInstances, 5, new Random(1));
		
		System.out.println(dataInstances.numAttributes());
		System.out.println(dataInstances.numInstances());
		System.out.println(dataInstances.numDistinctValues(0));
		System.out.println(dataInstances.attributeStats(0).missingCount);
		
		File file  = new File(args[1]);
		try {
			file.createNewFile();
			long endTime= System.nanoTime();
			FileWriter writer = new FileWriter(file);
			writer.write((endTime-startTime)/1000+" milisegundu \n");
			writer.write(args[1]+"\n");
			writer.write(evaluation.toMatrixString()+"\n");
			writer.write(evaluation.pctCorrect()+"\n");
			writer.write(evaluation.weightedFMeasure()+"\n");
			writer.write(evaluation.toSummaryString()+"\n");
			writer.flush();
			writer.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}

}
