require 'csv'

# Parameters
NUM_ITERATION = 1000
NUM_INPUT_DATA = 5
NUM_INPUT_LAYER = 10
NUM_HIDDEN_LAYER = 15
NUM_OUTPUT_LAYER = 5
LEARNING_RATE = 0.01

# Input Data
data = CSV.open("data.csv", "r")
line_num = 0
input_data = []
expected_result = []
data.each do |row|
    input_data << []
    field_cnt = 0
    NUM_INPUT_LAYER.times do |j|
        input_data[line_num] << row[field_cnt].to_i
        field_cnt += 1
    end
    expected_result << []
    NUM_OUTPUT_LAYER.times do |o|
        expected_result[line_num] << row[field_cnt].to_i
        field_cnt += 1
    end
    line_num += 1
end

# Input Layer Initialize
input_layer = []
NUM_INPUT_LAYER.times do |i|
    input_layer << 0.0
end

# Hidden Layer Initialize
hidden_layer = []
NUM_HIDDEN_LAYER.times do |i|
    hidden_layer << 0.0
end

# Output Layer Initialize
output_layer = []
NUM_INPUT_DATA.times do |d|
    output_layer << []
    NUM_OUTPUT_LAYER.times do |i|
        output_layer[d] << 0.0
    end
end

# Initialize Parameter
weight_input_hidden = []
NUM_INPUT_LAYER.times do |i|
    weight_input_hidden << []
    NUM_HIDDEN_LAYER.times do |h|
        weight_input_hidden[i] << rand
    end
end

weight_hidden_output = []
NUM_HIDDEN_LAYER.times do |i|
    weight_hidden_output << []
    NUM_OUTPUT_LAYER.times do |h|
        weight_hidden_output[i] << rand
    end
end

threshold_hidden = []
NUM_HIDDEN_LAYER.times do |i|
    threshold_hidden << rand
end

# Sigmoid Function
def sigmoid(val)
    return 1.0/(1.0 + Math.exp(-val))
end

def sigmoid_diff(val)
    return sigmoid(val)*(1.0 - sigmoid(val))
end

def output_layers_input(output_id, weight_hidden_output, hidden_layer)
    sum = 0.0
    NUM_HIDDEN_LAYER.times do |h|
        sum += weight_hidden_output[h][output_id] * hidden_layer[h]
    end
    return sum
end

def hidden_layers_input(hidden_id, weight_input_hidden, input_layer, threshold_hidden)
    sum = 0.0
    NUM_INPUT_LAYER.times do |i|
        sum += weight_input_hidden[i][hidden_id] * input_layer[i]
    end
    sum += threshold_hidden[hidden_id]
end

# Main Loop
NUM_ITERATION.times do |itr|
    #==========================
    # Calculate NN Output
    #==========================
    error = 0.0
    NUM_INPUT_DATA.times do |d|
        # Set data to input layer
        NUM_INPUT_LAYER.times do |i|
            input_layer[i] = input_data[d][i]
        end

        # Calculate hidden layer's output
        NUM_HIDDEN_LAYER.times do |h|
            sum = 0.0
            NUM_INPUT_LAYER.times do |i|
                sum += weight_input_hidden[i][h] * input_layer[i]
            end
            sum += threshold_hidden[h]
            hidden_layer[h] = sigmoid(sum)
        end

        # Calculate output layer's output
        NUM_OUTPUT_LAYER.times do |o|
            sum = 0.0
            NUM_HIDDEN_LAYER.times do |h|
                sum += weight_hidden_output[h][o] * hidden_layer[h]
            end
            output_layer[d][o] = sigmoid(sum)
        end

        # Calculated total error
        NUM_OUTPUT_LAYER.times do |o|
            error += (expected_result[d][o] - output_layer[d][o])*(expected_result[d][o] - output_layer[d][o])
        end
    end

    # print total error
    puts "Iteration #{itr+1} total error : #{error}"


    #=================
    # Learning
    #=================
    NUM_HIDDEN_LAYER.times do |h|
        NUM_INPUT_LAYER.times do |i|
            sum = 0.0
            NUM_INPUT_DATA.times do |d|
                tmp = 0.0
                NUM_OUTPUT_LAYER.times do |o|
                    tmp += (expected_result[d][o] - output_layer[d][o])*sigmoid_diff(output_layers_input(o, weight_hidden_output, hidden_layer))*weight_hidden_output[h][o]
                end
                tmp *= sigmoid_diff(hidden_layers_input(h, weight_input_hidden, input_layer, threshold_hidden))*input_data[d][i]
                sum += tmp
            end
            weight_input_hidden[i][h] += LEARNING_RATE * sum
        end
    end
    
    NUM_OUTPUT_LAYER.times do |o|
        NUM_HIDDEN_LAYER.times do |h|
            sum = 0.0
            NUM_OUTPUT_LAYER.times do |o|
                sum += (expected_result[0][o] - output_layer[0][o])*sigmoid_diff(output_layers_input(o, weight_hidden_output, hidden_layer))*hidden_layer[h]
            end
            weight_hidden_output[h][o] += LEARNING_RATE * sum
        end
    end

    NUM_HIDDEN_LAYER.times do |h|
        sum = 0.0
        NUM_INPUT_DATA.times do |d|
            tmp = 0.0
            NUM_OUTPUT_LAYER.times do |o|
                tmp += (expected_result[d][o] - output_layer[d][o])*sigmoid_diff(output_layers_input(o, weight_hidden_output, hidden_layer))*weight_hidden_output[h][o]
            end
            tmp *= sigmoid_diff(hidden_layers_input(h, weight_input_hidden, input_layer, threshold_hidden))
            sum += tmp
        end
        threshold_hidden[h] += LEARNING_RATE * sum
    end

end
