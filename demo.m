clear;
clc;

%urlwrite('http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', 'car.data');
data = importdata('iris_knn.txt');
%data = importdata('lenses.data.txt')
%[data, label] = readData(original_data);
%[train_data,train_label,test_data,test_label] = split(data,label);
%--------------------------------------------------------------------
[m,n] = size(data);
indices = randperm(m);data = data(indices,:); 
%above can be used to randomise the data-set
%% partition training and testing matrix
training = ceil(0.7*m);    testing = m - training;
trainingMatrix = data(1:training,1:n) ; testingMatrix = data(training+1:m,1:n);
%trainingMatrixSize = size(trainingMatrix,1);
%testingMatrixSize = size(testingMatrix,1);
kvalues = [5 10 20];
x = trainingMatrix(:,1:4);
y = trainingMatrix(:,5);

xmean= mean (x);
xstd = std(x);
x = zscore(x);
xtest = testingMatrix(:,1:4);
ytest = testingMatrix(:,5);

for i=1:testing
    xtest(i,:) = (xtest(i,:)- xmean)./ xstd;
end
%KNN Training Accuracy
fprintf('KNN Training results:\n');
for k = kvalues
    fprintf('For k = %d :\n',k);
    l1Accuracy = zeros(training,1);
    l2Accuracy = zeros(training,1);

for i= 1:training
    subtractedRows = zeros (size(x));
     for j=1:training
     subtractedRows(j,:) = x(j,:) - x(i,:);
     end
     norm1 = sum(abs(subtractedRows),2);
        norm1(i) = inf;
        
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm1Index] = min(norm1);
            kNearestNeighbours(krep) = y(minNorm1Index);
            norm1(minNorm1Index) = inf;
        end
      l1Accuracy(i) = (mode(kNearestNeighbours) == y(i)); 

        norm2 = sqrt(sum(subtractedRows.^2,2));
        norm2(i) = inf;
        
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm2Index] = min(norm2);
            kNearestNeighbours(krep) = y(minNorm2Index);
            norm2(minNorm2Index) = inf;
        end
       l2Accuracy(i) = (mode(kNearestNeighbours) == y(i));   
end
    l1Accuracy = sum(l1Accuracy)*100/training;
    l2Accuracy = sum(l2Accuracy)*100/training;

    fprintf('L1 training accuracy = %4.2f\n', l1Accuracy);
    fprintf('L2 training accuracy = %4.2f\n\n', l2Accuracy)
end

%KNN Testing accuracy
fprintf('KNN Testing results:\n');
for k = kvalues
    fprintf('For k = %d :\n',k);
    l1Accuracy = zeros(testing,1);
    l2Accuracy = zeros(testing,1);

    for i=1:testing
        subtractedRows = zeros(size(x));
        for j=1:training
            subtractedRows(j,:) = x(j,:) - xtest(i,:);
        end

        norm1 = sum(abs(subtractedRows),2);
        norm1(i) = inf;
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm1Index] = min(norm1);
            kNearestNeighbours(krep) = y(minNorm1Index);
            norm1(minNorm1Index) = inf;
        end
        l1Accuracy(i) = (mode(kNearestNeighbours) == ytest(i)); 

        
        norm2 = sqrt(sum(subtractedRows.^2,2));
        norm2(i) = inf;
        
        kNearestNeighbours = [];
        for krep=1:k
            [~,minNorm2Index] = min(norm2);
            kNearestNeighbours(krep) = y(minNorm2Index);
            norm2(minNorm2Index) = inf;
        end
        l2Accuracy(i) = (mode(kNearestNeighbours) == ytest(i));
    end

    l1Accuracy = sum(l1Accuracy)*100/testing;
    l2Accuracy = sum(l2Accuracy)*100/testing;

    fprintf('L1 testing accuracy = %4.2f\n', l1Accuracy);
    fprintf('L2 testing accuracy = %4.2f\n\n', l2Accuracy);
end