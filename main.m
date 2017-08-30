function main()
    clear all;
    close all;
    clc;

    global hidden;
    global dataset;
    global alpha;
    
    hidden = input('How many hidden neurons? ');
    if( isempty(hidden)); hidden = 4; end
    
    alpha = input('What should alpha be? ');
    if( isempty(alpha)); alpha = 0.2; end
    firstAlpha = alpha;
    
    count = 0;  
    
    fid = fopen('cancer.dt', 'r');
    while ~feof(fid)
          count = count + 1;
          line = fgets(fid);
          dataset(:,count) = sscanf(line, '%f %f %f %f %f %f %f %f %f %f %f');
    end
    fclose(fid);
    
    startNN(firstAlpha);
end

function startNN(firstAlpha)
    global hidden;
    global alpha;

    avemse = 0;
    avece = 0;
    
    %Do it a few times for nice stats
    for i=1:3
        nn.output.weights = [];
        nn.output.bias = 1;
        nn.output.sum = [];
        nn.output.inputs = [];
        nn.output.final = 0;

        for j=1:hidden
            nn.hiddenNodes(j) = node();
        end
        
        figure
        alpha = firstAlpha;
        nn = train(nn, i);
        avemse = avemse + nn.smse;
        avece = avece + nn.ce;
    end
    
    avemse = avemse/20;
    avece = avece/20;
    
    display(sprintf('\n-----\nAverage Mean Sqaure Error = %1.2f\nAverage Class Error = %1.2f\n', avemse, avece));
    
end

function nn = train(nn, sample)
    global hidden;
    global dataset;
    global alpha;
    
    %Generate weights for hidden nodes
    for i=1:hidden
        for j=1:9 %For each of the columns in the dataset
            nn.hiddenNodes(i).weights(j) = weightCalc();
        end
        nn.hiddenNodes(i).bias = 1;
    end
    
    %Generate weights for output neuron
    for j=1:hidden
        nn.output.weights(j) = weightCalc();
    end
    nn.output.bais = weightCalc();
    
    correct = 0;
    incorrect = 0;
    stopped = 0;
    iterations = 0;
    
    while (~stopped)
        iterations = iterations + 1;
        
        %Begin training
        terror = 0;
        for dataindex=1:350
            nn = forwardPass(nn, dataindex);
            validity = dataset(10,dataindex) - nn.output.final;
            if( validity == 0) %Then it's a hit
                correct = correct + 1; else incorrect = incorrect + 1;
            end
            nn = backwardPass(nn, dataindex);
            terror = terror + validity^2;
        end

        %Test network with validation set
        verror = 0;
        for dataindex=351:525
            nn = forwardPass(nn, dataindex);
            validity = dataset(10,dataindex) - nn.output.final;
            if( validity == 0) %Then it's a hit
                correct = correct + 1; else incorrect = incorrect + 1;
            end
            verror = verror + validity^2;
        end
        
        %Final test of the network
        ferror = 0;
        for dataindex=526:699
            nn = forwardPass(nn, dataindex);
            validity = dataset(10,dataindex) - nn.output.final;
            if( validity == 0) %Then it's a hit
                correct = correct + 1; else incorrect = incorrect + 1;
            end
            ferror = ferror + validity^2;
        end
        
        %if( iterations>1)     
        %    if( verror > old_vmse(iterations-1)+6) %If the old error is greater than the new one stop
        %        stopped = 1;
        %    end
            if( iterations>10)
                stopped = 1;
            end
       % end
       % if( alpha >0.002)
       %     alpha = alpha / 1.2; ** If we reduce alpha each epoch, more
       %     fine-grained adjustments are applied **
       % end
        old_tmse(iterations) = terror/350*100;
        old_vmse(iterations) = verror/175*100;
        old_fmse(iterations) = ferror/174*100;
        vect(iterations) = iterations;
        
        hold on;
        plot(vect, old_tmse, 'r');
        plot(vect, old_vmse, 'b');
        plot(vect, old_fmse, 'g');
        xlabel('Number of Epochs');
        ylabel('Error Rate (%)');
        title('Results generated from training a network on cancer data for 10 epochs');
        legend('Errors during training','Errors during validation','Errors during testing');
        drawnow;
        hold off;
    end
    
    smse = (old_tmse(iterations) + old_vmse(iterations) + old_fmse(iterations))/3;
    ce = incorrect/(incorrect+correct)*100;
    display(sprintf('Sample[%d]\nClass Error: %1.1f\nMean Square Error: %1.1f\n---\n', sample, ce, smse));
    nn.smse = smse;
    nn.ce = ce;
end

function weight = weightCalc()
    global hidden;
    min = -1 / sqrt(hidden);
    max = 1 / sqrt(hidden);
    for i=1:9
        weight = min + rand(1) * (max-min);
    end   
end


function fx = sigmoid(x)
    %e = exp(-2*x);
    %fx = (1-e)/(1+e); % better
    e = exp(-x);
    fx = 1/(1+e);
end

function dfx = invSigmoid(x)
    %e = exp(-2*x);
    %dfx = 1-((1-e)/(1+e))^2;
    e = exp(-x);
    fx = 1/(1+e);
    dfx = (fx)*(1 - (fx));
end

function nn = forwardPass(nn, dataindex)
    global hidden;
    global dataset;
    
    %Sum hidden layer
    for i=1:hidden
        nn.hiddenNodes(i).sum = sum( nn.hiddenNodes(i).weights * dataset(1:9,dataindex) + nn.hiddenNodes(i).bias);
        nn.output.inputs(i) = sigmoid( nn.hiddenNodes(i).sum);
    end
    
    nn.output.sum = sum( nn.output.weights .* nn.output.inputs) + nn.output.bias;
    nn.output.final = round( sigmoid( nn.output.sum));
    %display( sprintf('I[%d] %f %d', dataindex, sigmoid( nn.output.sum), nn.output.final));
end

function nn = backwardPass(nn, dataindex)
    global hidden;
    global dataset;
    global alpha;

    d1 = (dataset(10,dataindex) - sigmoid(nn.output.sum)) * invSigmoid(nn.output.sum);    
    for i=1:hidden
        temp = alpha * d1;
        nn.output.weights(i) = nn.output.weights(i) + (nn.output.inputs(i) * temp); 
        d2(i) = invSigmoid(nn.hiddenNodes(i).sum) * (nn.output.weights(i) * d1);
    end
    
    for i=1:hidden
        for j=1:9
            nn.hiddenNodes(i).weights(j) = nn.hiddenNodes(i).weights(j) + (alpha * dataset(j,dataindex) * d2(i));
        end
    end
end