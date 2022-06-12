import TextField  from '@mui/material/TextField';
import { Dispatch, SetStateAction } from 'react'

interface ParameterProps {
    parameterName: string
    defaultValue: number
    valueSetter: Dispatch<SetStateAction<number>>;
}

const ParameterWrapper = (props: ParameterProps) => {
  return (
    <TextField
      label={props.parameterName}
      type="number"
      onChange={(e) => {
        props.valueSetter(Number(e.target.value));
      }}
      variant="outlined"
      defaultValue={props.defaultValue}
    />
  );
}

export default ParameterWrapper
