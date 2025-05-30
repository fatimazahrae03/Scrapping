import kfp
from kfp import dsl

@dsl.pipeline(
    name='Hello World Pipeline',
    description='Un pipeline de test simple.'
)
def hello_pipeline(name: str = "Aya"):
    op = dsl.ContainerOp(
        name='Say Hello',
        image='alpine',
        command=['echo'],
        arguments=[f'Hello {name}']
    )

# Compiler le pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(hello_pipeline, 'hello_pipeline.yaml')
