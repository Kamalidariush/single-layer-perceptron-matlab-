nb_epoch= 100;
l_rate= 1;  
    
matrix = [[1.00, 1.0, 0.10, 1.0],
				[1.00, 2.0, 0.20, 1.0],
				[1.00, 0.10, 0.30, 0.0],
				[1.00, 2.0, 0.30, 1.0],
				[1.00, 0.20, 0.40, 0.0],
				[1.00, 3.0, 0.40, 1.0],
				[1.00, 0.10, 0.50, 0.0],
				[1.00, 1.50, 0.50, 1.0],
			  	[1.00, 0.50, 0.60, 0.0],
			  	[1.00, 1.60, 0.70, 0.0]];
	weights = [	0.05,	0.10,  0.20		];
weights=train_weights(matrix,weights,nb_epoch,l_rate);
figure();
x=transpose(matrix(:,2));
y=transpose(matrix(:,3));
c=ones(10,3)
c(:,1)=transpose(matrix(:,4));
c(:,2)=0;
hold on;
scatter(x,y,50,c,'filled');
plotpc(weights(1,2:3),weights(1,1));
hold off;

            
function prediction =predict (inputs,weights)
    activation=0;
%     for i=1:size(inputs,2)
%         activation=activation+inputs(1,i)*weights(1,i);
%     end
    x=inputs .* weights;
    activation=sum(x);
    if activation >=0.0
        prediction=1;
    else
        prediction=0;
    end
end

function ac1 =accuracy (matrix,weights)
    num_correct=0;
    ac1=0.0;
    pred=zeros(size(matrix,1),1);
    for i=1:size(matrix,1)
        pred(i)=predict(matrix(i,1:end-1),weights);
        if pred(i)==matrix(i,end)
        num_correct=num_correct + 1.0;
        end  
    end
    res=zeros(10,2)
    res(:,1)=pred(:,:); 
    res(:,2)=matrix(:,end);
    res
    ac1=num_correct/size(matrix,1)
end

function weight =train_weights (matrix,weights,nb_epoch,l_rate)
    for k=1:nb_epoch
          cur_acc = accuracy(matrix,weights);
         if cur_acc==1
            k=nb_epoch;
         end
        
         for i=1:size(matrix,1)
             prediction = predict(matrix(i,1:end-1),weights);
             error= matrix(i,end)-prediction;
             for j=1:size(weights,2)
                 weights(1,j) = weights(1,j)+(l_rate*error*matrix(i,j)); 
             end
             
         end
         
            weight=weights
     
    end
end

    
