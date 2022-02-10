package github;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class holdOutZeroR {
	
	static long startTime = System.nanoTime();
	
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource(args[0]);
		Instances dataInstances = source.getDataSet();
		
		dataInstances.setClassIndex(dataInstances.numAttributes() - 1);
		
		Randomize filteRandomize = new Randomize();
		filteRandomize.setInputFormat(dataInstances);
		Instances randomDataInstances = Filter.useFilter(dataInstances, filteRandomize);
		
		
		RemovePercentage filteRemovePercentage = new RemovePercentage();
		filteRemovePercentage.setInputFormat(randomDataInstances);
		filteRemovePercentage.setPercentage(30);
		Instances traInstances = Filter.useFilter(randomDataInstances, filteRemovePercentage);
		
		filteRemovePercentage = new RemovePercentage();
		filteRemovePercentage.setInputFormat(randomDataInstances);
		filteRemovePercentage.setPercentage(30);
		filteRemovePercentage.setInvertSelection(true);
		Instances test = Filter.useFilter(randomDataInstances, filteRemovePercentage);
		
		ZeroR modelR = new ZeroR();
		modelR.buildClassifier(traInstances);
		Evaluation evaluation = new Evaluation(traInstances);
		evaluation.evaluateModel(modelR, test);
		
		System.out.println(dataInstances.numAttributes());
		System.out.println(dataInstances.numInstances());
		System.out.println(dataInstances.attributeStats(0).missingCount);
		System.out.println(dataInstances.numDistinctValues(0));
		
		File file = new File(args[1]);
		try {
			file.createNewFile();
			long endTime = System.nanoTime();
			FileWriter writer = new FileWriter(file);
			writer.write((endTime-startTime)/1000 + " milisegundu \n");
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