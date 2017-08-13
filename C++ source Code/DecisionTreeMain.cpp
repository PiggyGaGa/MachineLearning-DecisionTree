#include "DecitionFunctions.h"
int main()
{
	string fileAddress;
	cout << "请输入文件的地址：";
	cin >> fileAddress;
	//cout << sizeof(unsigned int) << endl;
	machinelearning::DecisionTree myDecisionTree;
	myDecisionTree.ReadTrainDataFile(fileAddress);
	myDecisionTree.root = myDecisionTree.BuildTree(myDecisionTree.trainDataMat, myDecisionTree.vectorAttr, "ID3");


	// cout << myDecisionTree.dataMat << endl;
	cv::FileStorage fs("out.yml", cv::FileStorage::WRITE);
	int t = int(myDecisionTree.vectorAttr.size());
	for (int i = 0; i < t; i++)
	{
		fs << "Attribute" << myDecisionTree.vectorAttr[i].Attribute;
		fs << "typeNum" << myDecisionTree.vectorAttr[i].typeNum;
		// fs << "AttribureValue";
		for (int j = 0; j < myDecisionTree.vectorAttr[i].AttributeValue.size(); j++)
		{
			fs << "value "<< myDecisionTree.vectorAttr[i].AttributeValue[j];
		}
	}
	fs << "dataMat" << myDecisionTree.trainDataMat;
	fs.release();
	//myDecisionTree.root = myDecisionTree.BuildTree(myDecisionTree.trainDataMat, myDecisionTree.vectorAttr, "ID3");
	vector<string> result;
	vector<vector<string>> predictedData;
	predictedData = myDecisionTree.ReadPredictedDataFile("pre.csv");
	result = myDecisionTree.Predicted(myDecisionTree.root, predictedData);
	cout << "预测的结果如下：" << endl;
	for (int i = 0; i < result.size(); i++)
	{
		cout << result[i] << endl;
	}
	system("pause");
	return 0;
}
