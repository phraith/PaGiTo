import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate"
import CustomObjectFieldTemplateSmall from "./CustomObjectFieldTemplateSmall"

import instrumentationSchema from "./instrumentation.schema.json"

const uiSchema: UiSchema = {
        beam: {
            "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
        },
        detector: {
            
            resolution: {
                "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
                width: {
                    "ui:widget": "range"
                },
                height: {
                    "ui:widget": "range"
                }
            },
            beamImpact: {
                "ui:ObjectFieldTemplate": CustomObjectFieldTemplateSmall,
                x: {
                    "ui:widget": "range"
                },
                y: {
                    "ui:widget": "range"
                }
            }
        }

    
};

const schema: any = instrumentationSchema


function InstrumentationForm(props: { callback: Function, formData: any }) {
    const changeHandler = (value: any) => {
        props.callback(value.formData);
    };

    return (
            <JSONForm
                children={true}
                schema={schema}
                formData={props.formData}
                uiSchema={uiSchema}
                onChange={(value) => { changeHandler(value) }}
                />
    );
}

export default InstrumentationForm;
