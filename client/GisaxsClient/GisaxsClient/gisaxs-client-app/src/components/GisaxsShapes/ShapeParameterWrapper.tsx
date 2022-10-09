import Box from '@mui/material/Box/Box';
import TextField from '@mui/material/TextField';
import { Dispatch, SetStateAction, useEffect, useState } from 'react'
import ParameterWrapper from './ParameterWrapper';

interface ShapeParameterProps {
  parameterName: string
  jsonCallback: any
  initialParameterConfig: any
}

const ShapeParameter = (props: ShapeParameterProps) => {

  const [mean, setMean] = useState(props.initialParameterConfig.mean);
  const [stddev, setStddev] = useState(props.initialParameterConfig.stddev);

  useEffect(() => {
    props.jsonCallback(
      {
        mean: mean,
        stddev: stddev,
      },
      props.parameterName
    );
  }, [mean, stddev]);

  return (
    <Box display="flex" sx={{paddingBottom: 1}}>
      <ParameterWrapper
        defaultValue={mean}
        valueSetter={setMean}
        parameterName={`${props.parameterName}-mean`}
      />
      <ParameterWrapper
        defaultValue={stddev}
        valueSetter={setStddev}
        parameterName={`${props.parameterName}-stddev`}
      />
    </Box>
  );
}

export default ShapeParameter
