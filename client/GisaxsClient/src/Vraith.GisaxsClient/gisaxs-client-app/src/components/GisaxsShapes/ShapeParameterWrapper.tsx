import Box from '@mui/material/Box/Box';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography/Typography';
import { Dispatch, SetStateAction, useEffect, useState } from 'react'
import ParameterWrapper from './ParameterWrapper';

interface ShapeParameterProps {
  parameterName: string
  jsonCallback: any
  initialParameterConfig: any
  isSimulation: boolean
}

const ShapeParameter = (props: ShapeParameterProps) => {

  const [meanLower, setMeanLower] = useState(props.initialParameterConfig.meanLower);
  const [meanUpper, setMeanUpper] = useState(props.initialParameterConfig.meanUpper);
  const [stddevLower, setStddevLower] = useState(props.initialParameterConfig.stddevLower);
  const [stddevUpper, setStddevUpper] = useState(props.initialParameterConfig.stddevUpper);

  useEffect(() => {
    props.jsonCallback(
      {
        meanLower: meanLower,
        meanUpper: meanUpper,
        stddevLower: stddevLower,
        stddevUpper: stddevUpper
      },
      props.parameterName
    );
  }, [meanLower, meanUpper, stddevLower, stddevUpper]);

  return (
    <Box display="flex" sx={{ flexDirection: "column" }}>
      <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>{props.parameterName}</Typography>
      {!props.isSimulation &&
        <Box display="flex" sx={{ paddingBottom: 1 }}>
          <ParameterWrapper
            defaultValue={meanLower}
            valueSetter={setMeanLower}
            parameterName={"mean-min"}
          />
          <ParameterWrapper
            defaultValue={meanUpper}
            valueSetter={setMeanUpper}
            parameterName={"mean-max"}
          />
          <ParameterWrapper
            defaultValue={stddevLower}
            valueSetter={setStddevLower}
            parameterName={"stddev-min"}
          />
          <ParameterWrapper
            defaultValue={stddevUpper}
            valueSetter={setStddevUpper}
            parameterName={"stddev-max"}
          />
        </Box>
      }
      {props.isSimulation &&
        <Box display="flex" sx={{ paddingBottom: 1 }}>
          <ParameterWrapper
            defaultValue={meanUpper}
            valueSetter={(value) => { setMeanUpper(value); setMeanLower(value); }}
            parameterName={"mean"}
          />
          <ParameterWrapper
            defaultValue={stddevUpper}
            valueSetter={(value) => { setStddevUpper(value); setStddevLower(value); }}
            parameterName={"stddev"}
          />
        </Box>
      }
    </Box>
  );
}

export default ShapeParameter
