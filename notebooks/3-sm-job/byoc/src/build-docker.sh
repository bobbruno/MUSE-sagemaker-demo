# The name of our algorithm
algorithm_name=muse-large-000003

echo "Building container ${algorithm_name}"

cd src 

# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

docker build -t "${algorithm_name}" .

account=$(aws sts get-caller-identity --query Account --output text)
echo "Account retrieved"
# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
echo "Region is ${region}"
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}"
echo "Repository will be ${fullname}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "Repository ${algorithm_name} being created"
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker tag "${algorithm_name}:latest" ${fullname}

docker push ${fullname}
 
echo "Pushed "${algorithm_name}:${tag}" to ${fullname}"

echo "SUCCESS"