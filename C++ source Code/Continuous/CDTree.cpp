#include "CDTree.hpp"
#include <cstdio>
#include <algorithm>
//#define DEBUG_H

CDTree::CDTree(int deepestTree, int max_bin, float thresholdInfoGain)
{
    this->root = new TreeNode;
    this->root->label = -1;
    this->root->parents = nullptr;  // root node parents is null ptr
    this->deepestTree = deepestTree;
    this->max_bin = max_bin;
    this->thresholdInfoGain = thresholdInfoGain;
}

CDTree::CDTree()
{
    this->root = new TreeNode;
    this->root->parents = nullptr;  // root node parents is null ptr
    this->deepestTree = 4;
    this->max_bin = 3;
    this->thresholdInfoGain = 0.03;
}

int CDTree::buildTree(const MatrixXf &trainX, const MatrixXi &trainY, const MatrixXf &validateX, const MatrixXi &validateY, string Algorithm)
{
    this->intervals = continous2discrete(trainX, trainY);  // cut continous data to interval bins; 
    if(trainX.rows() != trainY.rows())
    {
        cerr << "train data and train label have no equal dimensions" << endl;
        std::exit(0);
    }
    if (validateX.rows() != validateY.rows())
    {
        cerr << "valid data and valid label have no equal dimensions" << endl;
        std::exit(0);
    }
    if (validateX.cols() != trainX.cols())
    {
        cerr << "valid data and train data have no equal cols " << endl;
        std::exit(0);
    }
    if (Algorithm == "ID3")
    {
        vector<int> currentAttributeIndex;
        for (int i = 0; i < trainX.cols(); i++)
        {
            currentAttributeIndex.push_back(i);
        }
        this->root = AlgorithmID3(this->root, trainX, trainY, currentAttributeIndex);
        cutBranches(validateX, validateY);
    }
    else if (Algorithm == "C4.5")
    {
        this->root = AlgorithmC4_5(trainX, trainY);
        cutBranches(validateX, validateY);
    }
    else if (Algorithm == "CART")
    {
        this->root = AlgorithmCART(trainX, trainY);
        cutBranches(validateX, validateY);
    }
    else
    {
        cerr << "Algorithm name must be ID3,C4.5 or CART, other algorithms are not supported, the program has been stopped!";
        exit(0);
    }
    return 0;
}

int CDTree::buildTree(const MatrixXf &trainX, const MatrixXi &trainY, string Algorithm)
{
    cout << "begin split continuous value to discrete intervals" << endl << "..." << endl;
    this->trainX_Dimension = trainX.cols();
    this->intervals = continous2discrete(trainX, trainY);
    cout << "discrete intervals success generatived" << endl << "..." << endl;
    if(trainX.rows() != trainY.rows())
    {
        cerr << "train data and train label have no equal dimensions" << endl;
        std::exit(0);
    }
    if (Algorithm == "ID3")
    {
        vector<int> currentAttributeIndex;
        for (int i = 0; i < trainX.cols(); i++)
        {
            currentAttributeIndex.push_back(i);
        }
        cout << "begin using ID3 algorithm build decision tree" << endl << "..." << endl;
        this->root = AlgorithmID3(this->root, trainX, trainY, currentAttributeIndex);
        cout << "using ID3 algorithm build decision tree success" << endl << "..." << endl;
        //cutBranches(4);
    }
    else if (Algorithm == "C4.5")
    {
        this->root = AlgorithmC4_5(trainX, trainY);
        cutBranches(4);
    }
    else if (Algorithm == "CART")
    {
        this->root = AlgorithmCART(trainX, trainY);
        cutBranches(4);
    }
    else
    {
        cerr << "Algorithm name must be ID3,C4.5 or CART, other algorithms are not supported, the program has been stopped!";
        exit(0);
    }
    return 0;
}

/*
cut branches according to validate data's accuracy
*/
int CDTree::cutBranches(const MatrixXf &validateX, const MatrixXi &validateY)
{
    return -1;
}
/*
cut branches according to deepest of the tree locked before;
*/
int CDTree::cutBranches(int deepthTree)
{
    return -1;
}

/*
核心算法， 利用ID3算法进行分支
*/
CDTree::TreeNode* CDTree::AlgorithmID3(TreeNode *parients, const MatrixXf &trainX, const MatrixXi &trainY, vector<int> attributeIndexOfCurrentData)
{
    vector<float> informationGain;  //信息增益
    int maxIndex;  //最大信息增益的属性的索引
    informationGain = CalculateInfGain(trainX, trainY, attributeIndexOfCurrentData);
    maxIndex = FindMaxInformationGain(informationGain);
    if (trainX.cols() == 1)
    {  //属性只有一列
        //TreeNode* leaf = new TreeNode;
        parients->AttributeIndex = -1;
        parients->LeafNode = true;
        parients->label = chooseMostLabel(trainY);//最多的那个
        return parients;
    }
    else if (TheSameLabel(trainY))
    {
        //标签值都相同
        //TreeNode* leaf = new TreeNode;
        parients->label = trainY(0, 0);
        parients->LeafNode = true;
        parients->AttributeIndex = -1;
        return parients;
    }
    else
    {
        
        //对分支节点进行数据切分
        //进行递归生成下一个层的节点
        parients->LeafNode = false;
        parients->AttributeIndex = attributeIndexOfCurrentData[maxIndex];

        //去掉最大信息增益的那个index
        vector<std::pair<float, float>> attributeInterval = this->intervals[attributeIndexOfCurrentData[maxIndex]];
        vector<int> currentAttributeIndex;
        attributeIndexOfCurrentData.erase(attributeIndexOfCurrentData.begin() + maxIndex);
        currentAttributeIndex = attributeIndexOfCurrentData;
        //TreeNode* branchNode = new TreeNode;

        if (attributeInterval.size() <= 1)
        {
            // 如果 只有一个区间，代表着不用分支
            //去掉maxIndex 那一列的数据即可
            TreeNode *child = new TreeNode;
            child->label = -1;
            child->parents = parients;
            string flag = "single"; //只有一个区间
            std::pair<float, float> smallInterval(0, 0);
            std::pair<MatrixXf, MatrixXi> splitresult = splitData(trainX, trainY, smallInterval, maxIndex, flag);
            child = AlgorithmID3(child, splitresult.first, splitresult.second, currentAttributeIndex);
            parients->children.push_back(child);

        }
        else
        {
            for (size_t i = 0; i < attributeInterval.size(); i++)
            {
                
                std::pair<float, float> smallInterval = attributeInterval[i];
                std::pair<MatrixXf, MatrixXi> splitresult;  // split data and split label
                // 如果是在区间在最左边，只需要判断小于这个值即可
                if (i == 0)
                {
                    string flag = "left";
                    splitresult = splitData(trainX, trainY, smallInterval, maxIndex, flag);
                }
                // 如果在区间最右边
                else if (i == attributeInterval.size() - 1)
                {
                    string flag = "right";
                    //std::pair<MatrixXf, MatrixXi> splitresult;  // split data and split label
                    splitresult = splitData(trainX, trainY, smallInterval, maxIndex, flag);
                }
                else
                {
                    string flag = "interval";
                    //std::pair<MatrixXf, MatrixXi> splitresult;  // split data and split label
                    splitresult = splitData(trainX, trainY, smallInterval, maxIndex, flag);
                }
                if (splitresult.first.size() == 0 || splitresult.second.size() == 0)
                {
                    //如果没有符合在这个区间的数据的话，不生成孩子节点。
                    TreeNode *child = new TreeNode;
                    child->parents = parients;
                    child->AttributeIndex = -1;
                    child->label = chooseMostLabel(trainY);
                    child->LeafNode = true;
                    parients->children.push_back(child);
                    continue;
                }
                TreeNode *child = new TreeNode;
                child->parents = parients;
                child->label = -1;
                child->LeafNode = false;
                child = AlgorithmID3(child, splitresult.first, splitresult.second, currentAttributeIndex);
                parients->children.push_back(child);            
            }
            if (parients->children.size() == 0)
            {
                //如果没有生成任何孩子节点
                cerr << "该分支节点没有进行切分" << endl;
                parients->AttributeIndex = -1;
                parients->label = chooseMostLabel(trainY);
                parients->LeafNode = true;
            }    
        } 
           
        //返回分支节点
        return parients;
    }
}

vector<float> CDTree::CalculateInfGain(const MatrixXf &trainX, const MatrixXi &trainY, vector<int> attributeIndexOfCurrentData)
{
    //第三个参数表示当前矩阵的各个列的属性对应最开始矩阵的index
    //1. 计算整个矩阵的熵
    float EntD;  //整个矩阵的信息熵
    vector<float> probability;
    probability = getSampleProbability(trainY);
    EntD = calculateEntropy(probability);

    vector<float> Gain;  //每个属性的信息增益值，函数的返回值
   
    // 计算每一个属性的信息增益
    for (int i= 0; i < trainX.cols(); i++)
    {
        int currentIndex = attributeIndexOfCurrentData[i];
        vector<std::pair<float, float>> currentParameterInterval = this->intervals[currentIndex];
        float sumEntD_v = 0;
        // 2. 计算 每个属性的某个取值所有样本的信息熵
        for (size_t j = 0; j < currentParameterInterval.size(); j++)
        {
            string flag;
            std::pair<float, float> smallInterval = currentParameterInterval[j];
            if (j == 0)
            {
                flag == "left";
            }
            else if (j == currentParameterInterval.size() - 1)
            {
                flag == "right";
            }
            else 
            {
                flag == "interval";
            }
            MatrixXi D_v_label = findDvlabel(trainX, trainY, i, smallInterval, flag);
            float EntDv;
            vector<float> prob = getSampleProbability(D_v_label);
            EntDv = calculateEntropy(prob);
            sumEntD_v = sumEntD_v + (D_v_label.rows() * 1.0 / trainY.rows()) * EntDv;
        }
        Gain.push_back(EntD - sumEntD_v);  //压入信息增益的向量
    }
    return Gain;

}
/* 从所有样本数据中计算每种标签的所占比重

*/
vector<float> CDTree::removeColRepetitionValue(const MatrixXf &Col)
{
    vector<float> result;
    for (int i = 0; i < Col.size(); i++)
    {
        vector<float>::iterator ret;
        ret = std::find(result.begin(), result.end(), Col(i));
        if (ret == result.end())
        {
            result.push_back(Col(i));
        }
    }
    return result;
}
vector<float> CDTree::getSampleProbability(const MatrixXi &trainY)
{
    vector<float> probability;
    vector<std::pair<int, int>> labelAndNum;   // <label, label's num in dataY>
    for (int i = 0; i < trainY.size(); i++)
    {
        findExistandDealwith(trainY(i, 0), labelAndNum);
    }
    for (size_t i = 0; i < labelAndNum.size(); i++)
    {
        probability.push_back(labelAndNum[i].second * 1.0 / trainY.rows());
    }
    return probability;
}

float CDTree::calculateEntropy(vector<float> probability)
{
    float result = 0;
    for (size_t i = 0; i < probability.size(); i++)
    {
        result = result + probability[i] * log2(probability[i]);
    }
    return -1 * result;
}

bool CDTree::TheSameLabel(const MatrixXi &trainY)
{
    bool flag = true;
    for (int i = 1; i < trainY.size(); i++)
    {
        if (trainY(i) != trainY(i - 1))
        {
            flag = false;
            return flag;
        }
    }
    return flag;
}
/*
参数1：trainX  训练数据
参数2: trainY 训练标签
参数3：index 矩阵trainX的第 index列所对应的属性
参数4：small interval  目的就是找到第index列的数据中在这个区间里的那个样本的标签值
*/
MatrixXi CDTree::findDvlabel(const MatrixXf &trainX, const MatrixXi &trainY, int index, std::pair<float, float> smallInterval, string flag)
{
    vector<int> label;
    if (flag == "left")
    {
        for (int i = 0; i < trainX.rows(); i++)
        {
            if (trainX(i, index) <= smallInterval.first)
            {
                label.push_back(trainY(i, 0));
            }
        }
    }
    else if (flag == "right")
    {
        for (int i = 0; i < trainX.rows(); i++)
        {
            if (trainX(i, index) > smallInterval.second)
            {
                label.push_back(trainY(i, 0));
            }
        }
    }
    else
    {
        for (int i = 0; i < trainX.rows(); i++)
        {
            if (trainX(i, index) > smallInterval.first && trainX(i, index) <= smallInterval.second)
            {
                label.push_back(trainY(i, 0));
            }
        }

    }
    MatrixXi result(label.size(), 1);
    for (size_t i = 0; i < label.size(); i++)
    {
        result(i) = label[i];
    }
    return result;
}

int CDTree::chooseMostLabel(const MatrixXi &trainY)
{
    vector<std::pair<int, int>> labelAndNum;   // <label, label's num in dataY>
    for (int i = 0; i < trainY.size(); i++)
    {
        findExistandDealwith(trainY(i), labelAndNum);
    }
    int max = labelAndNum[0].second;
    int maxlabel = labelAndNum[0].first;
    for (size_t i = 0; i < labelAndNum.size(); i++)
    {
        if (labelAndNum[i].second > max)
        {
            maxlabel = labelAndNum[i].first;
        }
    }
    return maxlabel;
}
int CDTree::findExistandDealwith(int label, vector<std::pair<int, int>> &labelAndNum)
{
    for(size_t i = 0; i < labelAndNum.size(); i++)
    {
        if (labelAndNum[i].first == label)
        {
            labelAndNum[i].second++;
            return 0;
        }
    }
    std::pair<int, int> a(label, 1);
    labelAndNum.push_back(a);
    return 0;
}
// 将上一节点的数据根据分支条件切分成下一个分支所需要的数据
std::pair<MatrixXf, MatrixXi> CDTree::splitData(const MatrixXf &trainX, const MatrixXi &trainY, std::pair<float, float> smallInterval, int maxIndex, string flag)
{
    std::pair<MatrixXf, MatrixXi> result;
    if (flag == "single")
    { 
        //该属性只有一种取值，也就是该分支节点只有一个分支。
        MatrixXf newMat(trainX.rows(), trainX.cols() - 1);
        for (int i = 0; i < trainX.rows(); i++)
        {
            newMat.block(0, 0, trainX.rows(), maxIndex + 1) = trainX.block(0, 0, trainX.rows(), maxIndex + 1);
            newMat.block(0, maxIndex, trainX.rows(), trainX.cols() - maxIndex - 1) = trainX.block(0, maxIndex + 1, trainX.rows(), trainX.cols() - maxIndex - 1);
        }
        result.first = newMat;
        result.second = trainY;

        return result;
    }
    if (flag == "left")
    {
        MatrixXf newMat;
        MatrixXi newLabel;
        for (int i = 0; i < trainX.rows(); i++)
        {

            if (trainX(i, maxIndex) <= smallInterval.second)
            {

                MatrixXf suitRowInMatrix(1, trainX.cols() - 1);  //满足条件的那一行数据全部加载进去
                if (maxIndex == trainX.cols() - 1)
                {
                    suitRowInMatrix = trainX.block(i, 0, 1, maxIndex);
                }
                else
                {
                    suitRowInMatrix.block(0, 0, 1, maxIndex) = trainX.block(i, 0, 1, maxIndex);
                    suitRowInMatrix.block(0, maxIndex, 1, trainX.cols() - maxIndex - 1) = trainX.block(i, maxIndex, 1, trainX.cols() - maxIndex - 1);
                } 

                newMat.conservativeResize(newMat.rows() + 1, trainX.cols() - 1);
                newMat.row(newMat.rows() - 1) = suitRowInMatrix; 
                newLabel.conservativeResize(newLabel.rows() + 1, 1);
                newLabel(newLabel.rows() - 1, 0) = trainY(i, 0);  
            }
        }
        result.first = newMat;
        result.second = newLabel;
        return result;
    }
    else if (flag == "right")
    {
        MatrixXf newMat;
        MatrixXi newLabel;
        for (int i = 0; i < trainX.rows(); i++)
        {

            if (trainX(i, maxIndex) >= smallInterval.first)
            {
                MatrixXf suitRowInMatrix(1, trainX.cols() - 1);  //满足条件的那一行数据全部加载进去
                if (maxIndex == trainX.cols() - 1)
                {
                    suitRowInMatrix = trainX.block(i, 0, 1, maxIndex);
                }
                else
                {
                    suitRowInMatrix.block(0, 0, 1, maxIndex) = trainX.block(i, 0, 1, maxIndex);
                    suitRowInMatrix.block(0, maxIndex, 1, trainX.cols() - maxIndex - 1) = trainX.block(i, maxIndex, 1, trainX.cols() - maxIndex - 1);
                } 
                
                newMat.conservativeResize(newMat.rows() + 1, trainX.cols() - 1);
                newMat.row(newMat.rows() - 1) = suitRowInMatrix;
                newLabel.conservativeResize(newLabel.rows() + 1, 1);
                newLabel(newLabel.rows() - 1, 0) = trainY(i, 0);  
                
            }
        }
        result.first = newMat;
        result.second = newLabel;
        return result;
    }
    else
    {

        for (int i = 0; i < trainX.rows(); i++)
        {
            if (trainX(i, maxIndex) <= smallInterval.second && trainX(i, maxIndex) >= smallInterval.first)
            {
                
            }
        }
        return result;
    }
}

int CDTree::FindMaxInformationGain(vector<float> s)
{
    int maxIndex = 0;
    float max = s[0];
    for (size_t i = 1; i < s.size(); i++)
    {
        if (s[i] > max)
        {
            max = s[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}



CDTree::TreeNode* CDTree::AlgorithmC4_5(const MatrixXf &trainX, const MatrixXi &trainY)
{

    return nullptr;
}
CDTree::TreeNode* CDTree::AlgorithmCART(const MatrixXf &trainX, const MatrixXi &trainY)
{
    return nullptr;
}

vector<vector<std::pair<float, float>>> CDTree::continous2discrete(const MatrixXf &trainX, const MatrixXi &trainY)
{
    vector<vector<std::pair<float, float>>> allColsIntervals;
    for (int i = 0; i < trainX.cols(); i++)
    {
        //
        #ifdef DEBUG_H
        cout << "开始第" << i << "列数据的区间分割" << endl;
        #endif
        vector<float> eachAttributeCut; //KMeans 聚类前的分割点
        vector<float> newEachAttributeCut;  //K means聚类后的分割点
        vector<std::pair<float, float>> eachAttributeIntervals;
        IntervalTree *intervalroot = new IntervalTree;
        intervalroot->parients = nullptr;
        // MatrixXf iColRemoverepetition;
        vector<float> newCol = removeColRepetitionValue(trainX.col(i));
        std::sort(newCol.begin(), newCol.end());
        intervalroot = MultiwayPartitionGain(intervalroot, newCol, trainX, trainY, i, this->thresholdInfoGain);
        //from interval tree get real intervals storage into vector
        getIntervalsFromTreeStruct(intervalroot, eachAttributeCut);
        std::sort(eachAttributeCut.begin(), eachAttributeCut.end());  //从树形结构出来的是乱序，要排序
        destroyIntervalTree(intervalroot);
        if (int(eachAttributeCut.size()) > this->max_bin - 1)
        {
            KMeans<float>(eachAttributeCut, newEachAttributeCut, this->max_bin - 1);  //利用k-means 对区间进行聚类，构成新的区间序列
            std::sort(newEachAttributeCut.begin(), newEachAttributeCut.end());
        }
        else
        {
            newEachAttributeCut = eachAttributeCut;
        }
        //通过分割点确定小区间
        for(size_t i = 0; i < newEachAttributeCut.size(); i++)
        {
            if (i == 0)
            {
                eachAttributeIntervals.push_back(std::pair<float, float>(newEachAttributeCut[i], newEachAttributeCut[i]));
            }
            else
            {
                eachAttributeIntervals.push_back(std::pair<float, float>(newEachAttributeCut[i - 1], newEachAttributeCut[i]));
            }
        }
        size_t dimension = newEachAttributeCut.size();
        if (dimension > 0)
        {
            eachAttributeIntervals.push_back(std::pair<float, float>(newEachAttributeCut[dimension - 1], newEachAttributeCut[dimension - 1]));
        }
        
        allColsIntervals.push_back(eachAttributeIntervals);
    }
    return allColsIntervals;
}

/*
optimal multi-way partition using gain method
从一列连续的向量中 找到合适的区间分割
参数1：连续的向量
参数2：原始数据矩阵
参数3：原始数据标签 参数1和参数2 在计算信息增益时用到
参数4：选取的是对应的第index列的连续属性
参数5：一个阈值，在计算分割区间的最大信息增益时，如果一个分割的最大信息增益比该阈值还小，那么就不进行分割。
返回值：存储分割区间信息的一个指针，用来构建区间分割的二叉树

*/
CDTree::IntervalTree* CDTree::MultiwayPartitionGain(IntervalTree *parientsNode, const vector<float> &continuousVec, const MatrixXf &trainX, const MatrixXi &trainY, int index, float thresholdInfoGain)
{
    
    if(continuousVec.size() < 2) //如果数据少于2个
    {
        parientsNode->LeafNode = true;
        /*
        IntervalTree *node = new IntervalTree;
        node->LeafNode = true;
        node->leftTree = nullptr;
        node->rightTree = nullptr;
        //node->nodeInterval = std::pair<float, float>(sortVec(0), sortVec(sortVec.cols() - 1));*/
        return parientsNode;
    }
    else
    {
        float EntD;  //整个矩阵的信息熵
        vector<float> probability;
        probability = getSampleProbability(trainY);
        EntD = calculateEntropy(probability);  //整个矩阵的信息熵
         
        //下面做的工作是计算每个分割点的信息增益
        float bestCutValue = 0;  //最大的信息增益的切割点的取值
        vector<float> Ta;  //每两个取值的中值
        vector<float> Gain_Ta;  //每个切割点的信息增益
        for (size_t i = 0; i < continuousVec.size() - 1; i++)
        {
            Ta.push_back((continuousVec[i] + continuousVec[i+1]) / 2);
        }

        for (size_t i = 0; i < Ta.size(); i++)  //这个循环依次计算每个分割点的信息增益值
        {
            std::pair<float, float> smallInterval(Ta[i], Ta[i]);
            float sumEnt = 0; 
            //left interval
            std::pair<MatrixXf, MatrixXi> splitresult;
            splitresult = splitData(trainX, trainY, smallInterval, index, "left");
            vector<float> prob = getSampleProbability(splitresult.second);
            float Entleft = calculateEntropy(prob);
            sumEnt = sumEnt + splitresult.second.size() * 1.0 / trainY.size() * Entleft;

            //right interval
            splitresult = splitData(trainX, trainY, smallInterval, index, "right");
            prob = getSampleProbability(splitresult.second);
            float Entright = calculateEntropy(prob);
            sumEnt = sumEnt + splitresult.second.size() * 1.0 / trainY.size() * Entright;
            Gain_Ta.push_back(EntD - sumEnt);
        }
        // 找到最大的信息增益的切割点
        float maxGain = Gain_Ta[0];
        int maxIndex = 0;
        for (size_t i = 1; i < Gain_Ta.size(); i++)
        {
            if (Gain_Ta[i] > maxGain)
            {
                maxGain = Gain_Ta[i];
                maxIndex = i;
            }
        }
        //最佳分割的中点
        bestCutValue = Ta[maxIndex];

        //如果该向量已经不能再继续分割了，怎样判断不能再分割了?信息增益小于阈值
        //如果当前分割
        if(maxGain < thresholdInfoGain)
        {
            
            parientsNode->LeafNode = true;
            return parientsNode;
        }
        else  //还能继续分割
        {
            //根据最佳分割点将向量分成两部分
            
            int cutIndex;
             
            for (size_t i = 0; i < continuousVec.size(); i++)
            {
                if (continuousVec[i] > bestCutValue)
                {
                    cutIndex = i;
                    break;
                }
            }
            vector<float> leftVector(continuousVec.begin(), continuousVec.begin() + cutIndex);
            vector<float> rightVector(continuousVec.begin() + cutIndex, continuousVec.end());

            /*
            #ifdef DEBUG_H
            cout << "leftVector " << endl;
            for (size_t i = 0; i < leftVector.size(); i++)
            {
                cout << leftVector[i] << endl;
            }
            
            cout << "rightVector " << endl;
            for (size_t i = 0; i < rightVector.size(); i++)
            {
                cout << rightVector[i] << endl;
            }
            #endif
            */
              //分割点左右两边的向量。

            //继续分割该向量
            //IntervalTree *node = new IntervalTree;
            parientsNode->LeafNode = false;
            parientsNode->cutValue = bestCutValue;
            parientsNode->leftTree = new IntervalTree;
            parientsNode->rightTree = new IntervalTree;
            parientsNode->leftTree->parients = parientsNode;
            parientsNode->rightTree->parients = parientsNode;
            //node->nodeInterval = std::pair<float, float>(0, 0); //如果不是叶子节点，那么该节点的区间是[0, 0]; invalid
            parientsNode->leftTree = MultiwayPartitionGain(parientsNode->leftTree, leftVector, trainX, trainY, index, thresholdInfoGain);
            parientsNode->rightTree = MultiwayPartitionGain(parientsNode->rightTree, rightVector, trainX, trainY, index, thresholdInfoGain);
            return parientsNode;
        }
    }    
    return nullptr;
}

int CDTree::sortVectorXf(const VectorXf &vec, VectorXf &sorted_vec, VectorXi &ind)
{ 
    ind = VectorXi::LinSpaced(vec.size(), 0, vec.size()-1);
    //[0 1 2 3 ... N-1] 
    auto rule = [vec](int i, int j)->bool{ return vec(i) < vec(j); };//正则表达式，作为sort的谓词 
    std::sort(ind.data(), ind.data()+ind.size(), rule); 
    //data成员函数返回VectorXd的第一个元素的指针，类似于begin() 
    sorted_vec.resize(vec.size()); 
    for(int i = 0; i < vec.size(); i++)
    { 
        sorted_vec(i)=vec(ind(i)); 
    } 
    return 0;
}

int CDTree::getIntervalsFromTreeStruct(IntervalTree *Intervalroot, vector<float> &intervals)
{
    if (Intervalroot->LeafNode == true)
    {
        if (Intervalroot->parients == nullptr)
        {
            return 0;
        }
        if (Intervalroot->parients->LeafNode == false)
        {
            intervals.push_back(Intervalroot->parients->cutValue);
            Intervalroot->parients->LeafNode = true;
        }    
        return 0;
    }
    else
    {
        getIntervalsFromTreeStruct(Intervalroot->leftTree, intervals);
        //delete Intervalroot->leftTree;
        getIntervalsFromTreeStruct(Intervalroot->rightTree, intervals);
        //delete Intervalroot->rightTree;
    }
    return 0;
}
template<typename Type>
int CDTree::KMeans(vector<Type> vec, vector<Type> &newvec, int k)
{
    if (int(vec.size()) < k)
    {
        std::cerr << "数据量小于聚类数k" << std::endl;
        return - 1;
    }
    else if (int(vec.size()) == k)
    {
        newvec = vec;
        return 0;
    }
    // findMinandMaxValu
    Type minValue = vec[0], maxValue = vec[0];
    for (size_t i = 1; i < vec.size(); i++)
    {
        if (vec[i] < minValue)
        {
            minValue = vec[i];
        }
        if (vec[i] > maxValue)
        {
            maxValue = vec[i];
        }
    }

    //
    srand((int)time(0));
    vector<Type> centers;
    //随机产生中心点
    for(int i =0; i < k; i++)
    {
        float centeri = (rand() / (RAND_MAX + 0.0)) * (maxValue - minValue) + minValue;
        centers.push_back(centeri);
    }

    std::vector<int> classNum(vec.size(), -1);
    
    while(1)
    {
        //计算每个样本到中心点的距离
        vector<vector<float>> distance(vec.size());
        for(size_t i = 0; i < vec.size(); i++)
        {
            float minDistance = maxValue - minValue;
            for (int j = 0; j < k; j++)
            {
                float dst  = abs(vec[i] - centers[j]);
                if (dst < minDistance)
                {
                    minDistance = dst;
                    classNum[i] = j;
                }
                distance[i].push_back(dst);
            }
        }

        //更新中心点坐标
        vector<Type> newCenters(k);
        int breakFlag = 0;
        for (int i = 0; i < k; i++)
        {
            float x1 = 0;
            int totalNum = 0;
            for (size_t j = 0; j < vec.size(); j++)
            {   
                if (classNum[j] == i)
                {    
                    x1 += vec[j];
                    totalNum++;
                }
            }
            if (totalNum == 0)
            {
                newCenters[i] = centers[i];
            }
            else
            {
                x1 = x1 / totalNum;
                newCenters[i] = x1;
            }
            if(abs(centers[i] - newCenters[i]) < 0.0001)
            {
                breakFlag++;
            }           
        }
        if (breakFlag == k)
        {
            break;
        }
        else
        {
            centers = newCenters;
        }

    }
    
    newvec = centers;
    return 0;
}

MatrixXi CDTree::predict(const MatrixXf &testX)
{
    MatrixXi result(testX.rows(), 1);
    if (testX.cols() != this->trainX_Dimension)
    {
        cerr << "test input data dimension is not aligned with train data dimension" << endl;
        exit(0);
    }
    for (int i = 0; i < testX.rows(); i++)
    {
        result(i, 0) = predictTree(this->root, testX.row(i));
    }
    return result;
}

int CDTree::predictTree(TreeNode *node, const MatrixXf testX)
{
    
    //要首先判断是否为叶子节点，在找attributeInterval时会造成指针越界  找到[-1]的位置，使程序崩溃

    if (node->LeafNode)
    {
        return node->label;
    }
    else
    {
        vector<std::pair<float, float>> attributeInterval = this->intervals[node->AttributeIndex];
        if (attributeInterval.size() != node->children.size() )
        {
            cerr << "分支节点的分支数和该节点的区间数不同" << endl;
            exit(0);
        }
        //TreeNode *newroot;
        bool findflag = false;
        for (size_t i = 0; i < node->children.size(); i++)
        {
            //在区间内
            if (i == 0)
            {
                string flag = "left";
                if(testX(node->AttributeIndex) <= attributeInterval[i].first)
                {
                    //cout << testX(node->AttributeIndex) << endl;
                    //newroot = node->children[i];
                    findflag = true;
                    return predictTree(node->children[i], testX);
                    break;
                }
            }
            else if (i == node->children.size() - 1)
            {
                string flag = "right";
                if(testX(node->AttributeIndex) >= attributeInterval[i].second)
                {
                    //newroot = node->children[i];
                    findflag = true;
                    return predictTree(node->children[i], testX);
                    break;
                }
            }
            else
            {
                if(testX(node->AttributeIndex) > attributeInterval[i].first && testX(node->AttributeIndex) < attributeInterval[i].second)
                {
                    //newroot = node->children[i];
                    findflag = true;
                    return predictTree(node->children[i], testX);
                    break;
                }
            }
        }
        if (! findflag)
        {
            cerr << "预测取值不在该分支节点的任一一个区间内，程序应该是有问题了，" << endl;
            exit(0);
        }
        
    }
    return 0;  //这一步不会被执行。
}
int CDTree::destroyTree(TreeNode *root)
{
    if (root->LeafNode)
    {
        if (root->parents != nullptr)
        {
            root->parents->LeafNode = true;
            delete root;
            return 0;
        }
        else
        {
            delete root;
            return 0;
        }       
    }
    else
    {
        for (size_t i = 0; i < root->children.size(); i++)
        {
            destroyTree(root->children[i]);
        }
    }
    if (root->LeafNode)
    {
        if (root->parents != nullptr)
        {
            root->parents->LeafNode = true;
            delete root;
        }
        else
        {
            delete root;
        }       
    }
    return 0;
}

int CDTree::destroyIntervalTree(IntervalTree *root)
{
    
    if (root->LeafNode == true)
    {
        if (root->parients == nullptr)
        {
            delete root;
            return 0;
        }
        else
        {
            if (root->parients->LeafNode == false)
            {
                root->parients->LeafNode = true;
            }
            delete root;
        }
        return 0;
    }
    else
    {
        destroyIntervalTree(root->leftTree);
        //delete Intervalroot->leftTree;
        destroyIntervalTree(root->rightTree);
        //delete Intervalroot->rightTree;
    }
    return 0;

}
CDTree::~CDTree()
{
    destroyTree(this->root);
}