import Box from '@mui/material/Box/Box';
import { useEffect, useState } from 'react';
import ParameterWrapper from './ParameterWrapper';

interface LocationParameterWrapperProps {
  jsonCallback: any
  initialLocationsConfig: any
}

const LocationParameterWrapper = (props: LocationParameterWrapperProps) => {

  const [x, setX] = useState(props.initialLocationsConfig.x);
  const [y, setY] = useState(props.initialLocationsConfig.y);
  const [z, setZ] = useState(props.initialLocationsConfig.z);

  useEffect(() => {
    props.jsonCallback(
      [{
        x: x,
        y: y,
        z: z
      }],
      'locations'
    )
  }, [x, y, z]);

  return (
    <Box display="flex" sx={{paddingBottom: 1}}>
      <ParameterWrapper
        defaultValue={x}
        valueSetter={setX}
        parameterName="x"
      />
      <ParameterWrapper
        defaultValue={y}
        valueSetter={setY}
        parameterName="y"
      />
      <ParameterWrapper
        defaultValue={z}
        valueSetter={setZ}
        parameterName="z"
      />
    </Box>
  );
}

export default LocationParameterWrapper
