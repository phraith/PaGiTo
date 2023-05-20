import Box from '@mui/material/Box/Box';
import { useEffect, useState } from 'react'
import ParameterWrapper from './ParameterWrapper';

interface RefractionParameterWrapperProps {
  jsonCallback: any
  initialRefractionConfig: any
}

const RefractionParameterWrapper = (props: RefractionParameterWrapperProps) => {

  const [delta, setDelta] = useState(props.initialRefractionConfig.delta);
  const [beta, setBeta] = useState(props.initialRefractionConfig.beta);

  useEffect(() => {
    props.jsonCallback(
      {
        delta: delta,
        beta: beta,
      },
      "refraction"
    );
  }, [delta, beta]);

  return (
    <Box display="flex" gap={2}>
      <ParameterWrapper
        defaultValue={delta}
        valueSetter={setDelta}
        parameterName={"refraction-delta"}
      />
      <ParameterWrapper
        defaultValue={beta}
        valueSetter={setBeta}
        parameterName={"refraction-beta"}
      />
    </Box>
  );
}

export default RefractionParameterWrapper
