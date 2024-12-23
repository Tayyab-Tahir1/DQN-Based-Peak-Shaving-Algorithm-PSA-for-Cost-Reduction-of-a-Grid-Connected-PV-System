% Load the data from CSV files
data1 = readtable('DQN_output_v2.csv');
data2 = readtable('Original_data.csv');

% Extract data for July (Month 7)
july_data1 = data1(data1.Month == 7, :);
july_data2 = data2(data2.Month == 7, :);

% Extract data for December (Month 12)
dec_data1 = data1(data1.Month == 12, :);
dec_data2 = data2(data2.Month == 12, :);

% Create continuous hour series for plotting
july_data1.Hour = (1:height(july_data1))';
july_data2.Hour = (1:height(july_data2))';
dec_data1.Hour = (1:height(dec_data1))';
dec_data2.Hour = (1:height(dec_data2))';

% comparison values
total_july_bill_dqn = sum(july_data1.Bill);
fprintf('DQN July month Bills: %.2f\n', total_july_bill_dqn);
total_july_bill_sam = sum(july_data2.Bill);
fprintf('SAM July month Bills: %.2f\n', total_july_bill_sam);

total_dec_bill_dqn = sum(dec_data1.Bill);
fprintf('DQN December month Bills: %.2f\n', total_dec_bill_dqn);
total_dec_bill_sam = sum(dec_data2.Bill);
fprintf('SAM December month Bills: %.2f\n', total_dec_bill_sam);


% Plot for July
figure;
plot(july_data1.Hour, july_data1.Bill, 'r', 'DisplayName', 'DQN Output');
hold on;
plot(july_data2.Hour, july_data2.Bill, 'b', 'DisplayName', 'Original Data');
title('July - Bill Comparison');
xlabel('Hour');
ylabel('Bill');
legend;
grid on;
hold off;

% Plot for December
figure;
plot(dec_data1.Hour, dec_data1.Bill, 'r', 'DisplayName', 'DQN Output');
hold on;
plot(dec_data2.Hour, dec_data2.Bill, 'b', 'DisplayName', 'Original Data');
title('December - Bill Comparison');
xlabel('Hour');
ylabel('Bill');
legend;
grid on;
hold off;
